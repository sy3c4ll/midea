#include "eigen/Dense"
#include "midifile/MidiFile.h"
#include "rtmidi/RtMidi.h"
#include "json/json.hpp"
#include <future>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>

#ifdef __unix__
#include <unistd.h>
#define NULLDEVICE "/dev/null"
#elif defined _WIN64
#include <windows.h>
#define sleep(x) Sleep((x)*1000)
#define NULLDEVICE "nul"
#endif

using Eigen::MatrixXd;
using Eigen::VectorXd;
using nlohmann::json;
using smf::MidiFile;
using std::async;
using std::cerr;
using std::cout;
using std::endl;
using std::future;
using std::future_status;
using std::getline;
using std::ifstream;
using std::move;
using std::mutex;
using std::ofstream;
using std::setw;
using std::string;
using std::thread;
using std::to_string;
using std::vector;
using std::chrono::duration;

#define LOGFILE "log.txt"
#define ERRFILE "log.txt"
#define LAYER_NUM 3
#define IO_SIZE 4
#define H_SIZE 10
#define EPOCH_NUM 10000
#define PRL 8
#define BATCH_SIZE 100
#define EPOCH_SAMPLE_PERIOD 1
#define SAMPLE_SIZE 1000
#define MAX_BATCH_SIZE 100000
#define ACT_FUNC tanh
#define LEARNING_RATE 0.01
#define NOTE_ON 0x90
#define T_MIN_HARD 0
#define T_MAX_HARD 3
#define N_MIN_HARD 0x00
#define N_MAX_HARD 0x7f
#define A_MIN_HARD 0x01
#define A_MAX_HARD 0x7f
#define T_MIN_SOFT 0
#define T_MAX_SOFT 3
#define N_MIN_SOFT 0x00
#define N_MAX_SOFT 0x7f
#define A_MIN_SOFT 0x20
#define A_MAX_SOFT 0x7f
#define RAND_NOTE_PERIOD 50
#define MAX_FILE_NUM 200000
#define MIDI_OUT 1
#define TPQ 960
#define BPM 120

#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))

const string CWD = "./";
const string DATA_DIR = CWD + "data/";
const string MODEL_DIR = CWD + "model/";
const string SAMPLE_DIR = CWD + "sample/";
const string METADATA_FILE = "METADATA";

struct model {
  MatrixXd wxy[LAYER_NUM + 1], waa[LAYER_NUM];
  VectorXd b[LAYER_NUM + 1], a[LAYER_NUM];
};

MatrixXd map(MatrixXd M, double (*fn)(double));
model *mset_zero(model *x);
model *mset_random(model *x);
model *mop_add(model *x, model y);
model *mop_sub(model *x, model y);
model *mop_mult(model *x, double y);
model *mop_div(model *x, double y);
model *msv_load(model *x, string path);
model *msv_save(model *x, string path);
VectorXd predict(model *m, VectorXd x);
VectorXd encode(VectorXd x);
VectorXd decode(VectorXd x);
void play_midi(RtMidiOut *midiout, MidiFile midi);
MidiFile sample(model m, bool rand);
model grad_desc(model m, MidiFile midi, double *loss);
double train(model *m, string *spath, string *epath, mutex *lock);

int main(int argc, char **argv) {

#ifdef LOGFILE
  cout << "[*] LOGFILE defined, redirecting stdout to " << LOGFILE << endl;
  freopen(LOGFILE, "w", stdout);
#endif /* LOGFILE */
#ifdef ERRFILE
  cout << "[*] ERRFILE defined, redirecting stderr to " << ERRFILE << endl;
  freopen(ERRFILE, "w", stderr);
#endif /* ERRFILE */

  model *m = mset_random(new model);
  if (argc == 2)
    msv_load(m, MODEL_DIR + "model-" + argv[1] + ".json");
  cout << "[*] Model layers, weights and biases initialised" << endl;

  ifstream md(DATA_DIR + METADATA_FILE);
  if (md.is_open())
    cout << "[*] Metadata parsed from " << DATA_DIR + METADATA_FILE << endl;
  else {
    cerr << "[!] Unable to open metadata file, check if "
         << DATA_DIR + METADATA_FILE << " exists. Aborting..." << endl;
    exit(EXIT_FAILURE);
  }

  string path[MAX_FILE_NUM] = {};
  size_t batch_num = 0;

  while (getline(md, path[batch_num])) {
    path[batch_num] = DATA_DIR + path[batch_num];
    batch_num++;
  }
  cout << "[*] " << batch_num << " MIDI file paths found" << endl;

  RtMidiOut *midiout = NULL;

#ifdef MIDI_OUT
  try {
    midiout = new RtMidiOut;
    midiout->openPort(MIDI_OUT);
  } catch (RtMidiError &e) {
    cerr << "[!] Unable to open MIDI output ports, aborting..." << endl;
    exit(EXIT_FAILURE);
  }
  if (!midiout->isPortOpen()) {
    cerr << "[!] Unable to open MIDI output ports, aborting..." << endl;
    exit(EXIT_FAILURE);
  }
  cout << "[*] Connected to MIDI output port " << MIDI_OUT << endl;
#endif /* MIDI_OUT */

  future<double> loss[PRL] = {};
  size_t ref[PRL][3] = {};
  mutex lock;
  MidiFile sample_midi;
  thread player;

  for (size_t epoch = argc == 2 ? (size_t)atoi(argv[1]) : 0; epoch < EPOCH_NUM;
       epoch++) {

    if (epoch % EPOCH_SAMPLE_PERIOD == 0) {

      msv_save(m, MODEL_DIR + "model-" + to_string(epoch) + ".json");
      cout << "[" << epoch << "/" << EPOCH_NUM << "] Model saved to "
           << MODEL_DIR << "model-" << epoch << ".json" << endl;

      sample_midi = sample(*m, false);
      cout << "[" << epoch << "/" << EPOCH_NUM << "] MIDI sample generated with 'normal' strategy"
           << endl;

      sample_midi.write(SAMPLE_DIR + "sample_norm_" + to_string(epoch) + ".mid");
      cout << "[" << epoch << "/" << EPOCH_NUM << "] MIDI sample saved to "
           << SAMPLE_DIR << "sample_norm_" << epoch << ".mid" << endl;

      sample_midi = sample(*m, true);
      cout << "[" << epoch << "/" << EPOCH_NUM << "] MIDI sample generated with 'random' strategy"
           << endl;

      sample_midi.write(SAMPLE_DIR + "sample_rand_" + to_string(epoch) + ".mid");
      cout << "[" << epoch << "/" << EPOCH_NUM << "] MIDI sample saved to "
           << SAMPLE_DIR << "sample_rand_" << epoch << ".mid" << endl;

      if (midiout != NULL) {
        player = thread(play_midi, midiout, sample_midi);
        player.detach();
        cout << "[" << epoch << "/" << EPOCH_NUM << "] Now playing: sample_rand_"<<epoch
             << endl;
      }
    }

    for (size_t loc = 0; loc + BATCH_SIZE < batch_num; loc += BATCH_SIZE)
      for (size_t thr = 0; thr < PRL; thr = (thr + 1) % PRL)
        if (!loss[thr].valid() ||
            loss[thr].wait_for(duration<int>(0)) == future_status::ready) {

          if (ref[thr][0] != 0)
            cout << "[" << ref[thr][0] << "/" << EPOCH_NUM
                 << "] Model trained with batch " << ref[thr][1] << "-"
                 << ref[thr][2] - 1 << "/" << batch_num << " with loss "
                 << loss[thr].get() << " on thread " << thr << endl;

          ref[thr][0] = epoch + 1;
          ref[thr][1] = loc;
          ref[thr][2] = loc + BATCH_SIZE;

          loss[thr] =
              async(train, m, path + ref[thr][1], path + ref[thr][2], &lock);

          break;
        }

    for (size_t thr = 0; thr < PRL; thr++)
      if (!loss[thr].valid() ||
          loss[thr].wait_for(duration<int>(0)) == future_status::ready) {

        if (ref[thr][0] != 0)
          cout << "[" << ref[thr][0] << "/" << EPOCH_NUM
               << "] Model trained with batch " << ref[thr][1] << "-"
               << ref[thr][2] - 1 << "/" << batch_num << " with loss "
               << loss[thr].get() << " from thread " << thr << endl;

        ref[thr][0] = epoch + 1;
        ref[thr][1] = batch_num - (batch_num % BATCH_SIZE);
        ref[thr][2] = batch_num;

        loss[thr] =
            async(train, m, path + ref[thr][1], path + ref[thr][2], &lock);

        break;
      }
  }

  msv_save(m, MODEL_DIR + "model-" + to_string(EPOCH_NUM) + ".json");
  cout << "[" << EPOCH_NUM << "/" << EPOCH_NUM << "] Model saved to "
       << MODEL_DIR << "model-" << EPOCH_NUM << ".json" << endl;

  sample_midi = sample(*m, false);
  cout << "[" << EPOCH_NUM << "/" << EPOCH_NUM << "] MIDI sample generated with 'normal' strategy"
       << endl;

  sample_midi.write(SAMPLE_DIR + "sample_norm_" + to_string(EPOCH_NUM) + ".mid");
  cout << "[" << EPOCH_NUM << "/" << EPOCH_NUM << "] MIDI sample saved to "
       << SAMPLE_DIR << "sample_norm_" << EPOCH_NUM << ".mid" << endl;

  sample_midi = sample(*m, true);
  cout << "[" << EPOCH_NUM << "/" << EPOCH_NUM << "] MIDI sample generated with 'random' strategy"
       << endl;

  sample_midi.write(SAMPLE_DIR + "sample_rand_" + to_string(EPOCH_NUM) + ".mid");
  cout << "[" << EPOCH_NUM << "/" << EPOCH_NUM << "] MIDI sample saved to "
       << SAMPLE_DIR << "sample_rand_" << EPOCH_NUM << ".mid" << endl;

  if (midiout != NULL) {
    player = thread(play_midi, midiout, sample_midi);
    player.join();
    cout << "[" << EPOCH_NUM << "/" << EPOCH_NUM << "] Now playing: sample 1"
         << endl;
  }

  cout << "[*] Model successfully trained over " << EPOCH_NUM
       << " epochs, exiting..." << endl;

  delete m;
  delete midiout;
  return EXIT_SUCCESS;
}

MatrixXd map(MatrixXd M, double (*fn)(double)) {

  for (size_t i = 0; i < (size_t)M.rows(); i++)
    for (size_t j = 0; j < (size_t)M.cols(); j++)
      M(i, j) = (*fn)(M(i, j));

  return M;
}

model *mset_zero(model *x) {

  for (size_t i = 0; i < LAYER_NUM; i++)
    x->a[i] = VectorXd::Zero(H_SIZE);

  for (size_t i = 1; i < LAYER_NUM; i++)
    x->wxy[i] = MatrixXd::Zero(H_SIZE, H_SIZE);
  x->wxy[0] = MatrixXd::Zero(H_SIZE, IO_SIZE);
  x->wxy[LAYER_NUM] = MatrixXd::Zero(IO_SIZE, H_SIZE);

  for (size_t i = 0; i < LAYER_NUM; i++)
    x->waa[i] = MatrixXd::Zero(H_SIZE, H_SIZE);

  for (size_t i = 0; i < LAYER_NUM; i++)
    x->b[i] = VectorXd::Zero(H_SIZE);
  x->b[LAYER_NUM] = VectorXd::Zero(IO_SIZE);

  return x;
}

model *mset_random(model *x) {

  for (size_t i = 0; i < LAYER_NUM; i++)
    x->a[i] = VectorXd::Random(H_SIZE);

  for (size_t i = 1; i < LAYER_NUM; i++)
    x->wxy[i] = MatrixXd::Random(H_SIZE, H_SIZE);
  x->wxy[0] = MatrixXd::Random(H_SIZE, IO_SIZE);
  x->wxy[LAYER_NUM] = MatrixXd::Random(IO_SIZE, H_SIZE);

  for (size_t i = 0; i < LAYER_NUM; i++)
    x->waa[i] = MatrixXd::Random(H_SIZE, H_SIZE);

  for (size_t i = 0; i < LAYER_NUM; i++)
    x->b[i] = VectorXd::Random(H_SIZE);
  x->b[LAYER_NUM] = VectorXd::Random(IO_SIZE);

  return x;
}

model *mop_add(model *x, model y) {

  for (size_t i = 0; i < LAYER_NUM + 1; i++)
    x->wxy[i] += y.wxy[i], x->b[i] += y.b[i];
  for (size_t i = 0; i < LAYER_NUM; i++)
    x->waa[i] += y.waa[i], x->a[i] += y.a[i];
  return x;
}

model *mop_sub(model *x, model y) {

  for (size_t i = 0; i < LAYER_NUM + 1; i++)
    x->wxy[i] -= y.wxy[i], x->b[i] -= y.b[i];
  for (size_t i = 0; i < LAYER_NUM; i++)
    x->waa[i] -= y.waa[i], x->a[i] -= y.a[i];
  return x;
}

model *mop_mult(model *x, double y) {

  for (size_t i = 0; i < LAYER_NUM + 1; i++)
    x->wxy[i] *= y, x->b[i] *= y;
  for (size_t i = 0; i < LAYER_NUM; i++)
    x->waa[i] *= y, x->a[i] *= y;
  return x;
}

model *mop_div(model *x, double y) {

  for (size_t i = 0; i < LAYER_NUM + 1; i++)
    x->wxy[i] /= y, x->b[i] /= y;
  for (size_t i = 0; i < LAYER_NUM; i++)
    x->waa[i] /= y, x->a[i] /= y;
  return x;
}

model *msv_load(model *x, string path) {

  json sv = json::parse(ifstream(path));
  for (size_t i = 0; i < LAYER_NUM + 1; i++)
    for (size_t j = 0; j < (size_t)x->wxy[i].rows(); j++)
      for (size_t k = 0; k < (size_t)x->wxy[i].cols(); k++)
        x->wxy[i](j, k) = sv["wxy"][i][j][k];
  for (size_t i = 0; i < LAYER_NUM; i++)
    for (size_t j = 0; j < (size_t)x->waa[i].rows(); j++)
      for (size_t k = 0; k < (size_t)x->waa[i].cols(); k++)
        x->waa[i](j, k) = sv["waa"][i][j][k];
  for (size_t i = 0; i < LAYER_NUM + 1; i++)
    for (size_t j = 0; j < (size_t)x->b[i].rows(); j++)
      x->b[i](j) = sv["b"][i][j];
  for (size_t i = 0; i < LAYER_NUM; i++)
    for (size_t j = 0; j < (size_t)x->a[i].rows(); j++)
      x->a[i](j) = sv["a"][i][j];
  return x;
}

model *msv_save(model *x, string path) {

  nlohmann::json sv;
  for (size_t i = 0; i < LAYER_NUM + 1; i++)
    for (size_t j = 0; j < (size_t)x->wxy[i].rows(); j++)
      for (size_t k = 0; k < (size_t)x->wxy[i].cols(); k++)
        sv["wxy"][i][j][k] = x->wxy[i](j, k);
  for (size_t i = 0; i < LAYER_NUM; i++)
    for (size_t j = 0; j < (size_t)x->waa[i].rows(); j++)
      for (size_t k = 0; k < (size_t)x->waa[i].cols(); k++)
        sv["waa"][i][j][k] = x->waa[i](j, k);
  for (size_t i = 0; i < LAYER_NUM + 1; i++)
    for (size_t j = 0; j < (size_t)x->b[i].rows(); j++)
      sv["b"][i][j] = x->b[i](j);
  for (size_t i = 0; i < LAYER_NUM; i++)
    for (size_t j = 0; j < (size_t)x->a[i].rows(); j++)
      sv["a"][i][j] = x->a[i](j);
  ofstream(path) << setw(2) << sv << endl;
  return x;
}

VectorXd predict(model *m, VectorXd x) {

  m->a[0] = map(m->wxy[0] * x + m->waa[0] * m->a[0] + m->b[0], ACT_FUNC);
  for (size_t i = 1; i < LAYER_NUM; i++)
    m->a[i] =
        map(m->wxy[i] * m->a[i - 1] + m->waa[i] * m->a[i] + m->b[i], ACT_FUNC);

  return map(m->wxy[LAYER_NUM] * m->a[LAYER_NUM - 1] + m->b[LAYER_NUM],
             ACT_FUNC);
}

VectorXd encode(VectorXd x) {

  x(0) = (x(0) - T_MIN_HARD) / (T_MAX_HARD - T_MIN_HARD) * 2 - 1;
  x(1) = (x(1) - T_MIN_HARD) / (T_MAX_HARD - T_MIN_HARD) * 2 - 1;
  x(2) = (x(2) - N_MIN_HARD) / (N_MAX_HARD - N_MIN_HARD) * 2 - 1;
  x(3) = (x(3) - A_MIN_HARD) / (A_MAX_HARD - A_MIN_HARD) * 2 - 1;
  return x;
}

VectorXd decode(VectorXd x) {

  x(0) = min(max((x(0) + 1) * (T_MAX_HARD - T_MIN_HARD) / 2 + T_MIN_HARD,T_MIN_SOFT),T_MAX_SOFT);
  x(1) = min(max((x(1) + 1) * (T_MAX_HARD - T_MIN_HARD) / 2 + T_MIN_HARD,T_MIN_SOFT),T_MAX_SOFT);
  x(2) = round(min(max((x(2) + 1) * (N_MAX_HARD - N_MIN_HARD) / 2 + N_MIN_HARD,N_MIN_SOFT),N_MAX_SOFT));
  x(3) = round(min(max((x(3) + 1) * (A_MAX_HARD - A_MIN_HARD) / 2 + A_MIN_HARD,A_MIN_SOFT),A_MAX_SOFT));
  return x;
}

void play_midi(RtMidiOut *midiout, MidiFile midi) {

  if (!midi.status()) {
    cerr << "[!] play_midi: Unable to process MIDI data from arguments" << endl;
    return;
  }

  midi.absoluteTicks();
  midi.joinTracks();
  midi.doTimeAnalysis();

  double psec = 0;

  for (size_t i = 0; i < (size_t)midi[0].size(); i++) {
    sleep(midi[0][i].seconds - psec);
    midiout->sendMessage(&midi[0][i]);
    psec = midi[0][i].seconds;
  }

  return;
}

MidiFile sample(model m, bool rand) {

  MidiFile midi;
  vector<uint8_t> note{NOTE_ON, 0, 0};
  VectorXd x(IO_SIZE), n(IO_SIZE);
  double atime = 0;
  int temp = 0;
  float randtemp = 0;
  midi.absoluteTicks();
  midi.setTPQ(TPQ);
  midi.addTrack(1);

  for (size_t i = 0; i < SAMPLE_SIZE; i++) {
    if (i)
      x = predict(&m, x);
    if (!i || (rand && !i % RAND_NOTE_PERIOD))
      x = VectorXd::Random(IO_SIZE);

    n = decode(x);

    note[1] = n(2), note[2] = n(3);
    midi.addEvent(1, (int)((atime + n(0)) * TPQ * BPM / 60), note);

    note[2] = 0;
    midi.addEvent(1, (int)((atime + n(0) + n(1)) * TPQ * BPM / 60), note);

    cout << "[*] sample: Recorded note " << (int)n(2) << " of attack "
         << (int)n(3) << " ON to " << atime + n(0) << " and OFF to "
         << atime + n(0) + n(1) << endl;

    atime += n(0);
  }

  midi.sortTracks();
  return midi;
}

model grad_desc(model m, MidiFile midi, double *loss) {

  model grad;
  mset_zero(&grad);
  uint8_t inst=0;

  if (!midi.status()) {
    cerr << "[!] Unable to parse MIDI data from argument, skipping..." << endl;
    *loss = 0;
    return grad;
  }
  midi.absoluteTicks();
  midi.joinTracks();
  midi.doTimeAnalysis();
  midi.linkNotePairs();
  size_t batch_size = 0;
  VectorXd *x = new VectorXd[MAX_BATCH_SIZE], *y = new VectorXd[MAX_BATCH_SIZE],
           *a[LAYER_NUM] = {}, d;
  for (size_t i = 0; i < LAYER_NUM; i++)
    a[i] = new VectorXd[MAX_BATCH_SIZE];
  double ptime = 0;
  *loss = 0;

  for (size_t i = 0; i < (size_t)midi[0].size(); i++) {

    if (midi[0][i].isPatchChange())
      inst=midi[0][i][1];

    if (midi[0][i].isNoteOn() && inst<80) {

      for (size_t j = 0; j < LAYER_NUM; j++)
        a[j][batch_size] = m.a[j];

      x[batch_size].resize(IO_SIZE);
      x[batch_size] << midi[0][i].seconds - ptime,
          midi[0][i].getDurationInSeconds(), midi[0][i][1], midi[0][i][2];
      x[batch_size] = encode(x[batch_size]);

      y[batch_size] = predict(&m, x[batch_size]);

      if (batch_size > 0)
        for (size_t j = 0; j < IO_SIZE; j++)
          *loss += (x[batch_size](j) - y[batch_size - 1](j)) *
                   (x[batch_size](j) - y[batch_size - 1](j)) / IO_SIZE;

      ptime = midi[0][i].seconds;

      batch_size++;
    }
  }

  if (batch_size == 0) {
    *loss = 0;
    return grad;
  } else
    *loss /= batch_size;

  for (size_t i = batch_size; i > 0; i--) {

    if (i > 1) {

      d = y[i - 2] - x[i - 1];
      for (size_t k = 0; k < IO_SIZE; k++)
        d(k) *= 1 - y[i - 2](k) * y[i - 2](k);
    }

    for (size_t j = LAYER_NUM; j > 0; j--) {

      if (i < batch_size) {
        for (size_t k = 0; k < H_SIZE; k++)
          grad.a[j - 1](k) *= 1 - a[j - 1][i](k) * a[j - 1][i](k);
        grad.waa[j - 1] = grad.a[j - 1] * a[j - 1][i - 1].transpose();
        grad.a[j - 1] = m.waa[j - 1].transpose() * grad.a[j - 1];
      }

      if (i > 1) {
        grad.wxy[j] += d * a[j - 1][i - 1].transpose();
        grad.b[j] += d;
        grad.a[j - 1] += m.wxy[j].transpose() * d;
      }

      d = grad.a[j - 1];
      for (size_t k = 0; k < H_SIZE; k++)
        d(k) *= 1 - a[j - 1][i - 1](k) * a[j - 1][i - 1](k);
    }

    grad.wxy[0] += d * x[i - 1].transpose();
    grad.b[0] += d;
  }

  delete[] x;
  delete[] y;
  for (size_t i = 0; i < LAYER_NUM; i++)
    delete[] a[i];

  return *mop_div(&grad, (double)batch_size);
}

double train(model *m, string *spath, string *epath, mutex *lock) {

  model grad;
  mset_zero(&grad);
  MidiFile midi;
  size_t batch_num = 0;
  double loss = 0, batch_loss = 0;

  for (string *path = spath; path < epath; path++)
    if (*path != "") {

      midi.read(*path);

      if (!midi.status()) {
        cerr << "[!] Invalid MIDI data, " << *path
             << " will be excluded from training" << endl;
        *path = "";
        continue;
      }

      mop_add(&grad, grad_desc(*m, midi, &batch_loss));
      loss += batch_loss;
      batch_num++;
    }

  lock->lock();
  mop_add(m, *mop_mult(mop_div(&grad, batch_num), -LEARNING_RATE));
  lock->unlock();

  loss /= batch_num;
  return loss;
}
