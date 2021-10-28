#include "eigen/Dense"
#include "midifile/MidiFile.h"
#include "rtmidi/RtMidi.h"
#include "json/json.hpp"
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
using std::cerr;
using std::cout;
using std::endl;
using std::getline;
using std::ifstream;
using std::ofstream;
using std::setw;
using std::string;
using std::thread;
using std::to_string;
using std::vector;

#define LAYER_NUM 3
#define IO_SIZE 4
#define H_SIZE 10
#define EPOCH_NUM 500
#define MINI_BATCH_SIZE 100
#define EPOCH_SAMPLE_PERIOD 10
#define SAMPLE_SIZE 200
#define MAX_BATCH_SIZE 100000
#define ACT_FUNC tanh
#define LEARNING_RATE 0.01
#define NOTE_ON 0x90
#define T_MIN 0
#define T_MAX 3
#define N_MIN 0x00
#define N_MAX 0x7f
#define A_MIN 0x01
#define A_MAX 0x7f
#define MAX_FILE_NUM 200000
#define MIDI_OUT 1
#define TPQ 960
#define BPM 120

const string DATA_DIR("./data/");
const string MODEL_DIR("./model/");
const string SAMPLE_DIR("./sample/");
const string METADATA_FILE("METADATA");

struct model {
  MatrixXd wxy[LAYER_NUM + 1], waa[LAYER_NUM];
  VectorXd b[LAYER_NUM + 1], a[LAYER_NUM];
};

inline double min(double x, double y) { return x < y ? x : y; }
inline double max(double x, double y) { return x > y ? x : y; }
inline double cap(double x) { return min(max(x, -1), 1); }

MatrixXd map(MatrixXd M, void *func);
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
MidiFile sample(model m, VectorXd *seq, int seqsize);
model grad_desc(model m, MidiFile *midi, double *loss);
void train(model *m, MidiFile *midi, size_t batch_size, size_t sepoch,
           RtMidiOut *midiout);

int main(int argc, char **argv) {

#ifdef LOGFILE
  cout << "[*] LOGFILE defined, redirecting output to " << LOGFILE << endl;
  freopen(LOGFILE, "w", stdout);
  freopen(LOGFILE, "w", stderr);
#endif /* LOGFILE */

  model *m = mset_random(new model);
  // msv_load(m, MODEL_DIR+"model-"+argv[1]+".json");

  cout << "[*] Model layers, weights and biases initialised" << endl;

  ifstream md(DATA_DIR + METADATA_FILE);
  if (md.is_open())
    cout << "[*] Metadata parsed from " << DATA_DIR + METADATA_FILE << endl;
  else {
    cerr << "[!] Unable to open metadata file, check if it exists. Exiting..."
         << endl;
    exit(EXIT_FAILURE);
  }

  MidiFile *midi = new MidiFile[MAX_FILE_NUM];
  string path("");
  size_t batch_num = 0;

  while (getline(md, path)) {

    path = DATA_DIR + path;
    midi[batch_num].read(path);

    if (!midi[batch_num].status()) {
      cerr << "[!] Unable to open MIDI file from path " << path << endl;
      continue;
    }

    batch_num++;
  }
  cout << "[*] " << batch_num << " MIDI files successfully loaded" << endl;

  RtMidiOut *midiout = NULL;

#ifdef MIDI_OUT
  try {
    midiout = new RtMidiOut();
    midiout->openPort(MIDI_OUT);
  } catch (RtMidiError &e) {
    cerr << "[!] Unable to open MIDI output ports, exiting..." << endl;
    exit(EXIT_FAILURE);
  }
  if (!midiout->isPortOpen()) {
    cerr << "[!] Unable to open MIDI output ports, exiting..." << endl;
    exit(EXIT_FAILURE);
  }
  cout << "[*] Connected to MIDI output port " << MIDI_OUT << endl;
#endif

  train(m, midi, batch_num, (size_t)atoi(argv[1]), midiout);

  cout << "[*] Model successfully trained over " << EPOCH_NUM
       << " epochs, quitting..." << endl;

  delete m;
  delete midiout;
  delete[] midi;
  return 0;
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

  x(0) = (x(0) - T_MIN) / (T_MAX - T_MIN) * 2 - 1;
  x(1) = (x(1) - T_MIN) / (T_MAX - T_MIN) * 2 - 1;
  x(2) = (x(2) - N_MIN) / (N_MAX - N_MIN) * 2 - 1;
  x(3) = (x(3) - A_MIN) / (A_MAX - A_MIN) * 2 - 1;
  return x;
}

VectorXd decode(VectorXd x) {

  x(0) = (x(0) + 1) * (T_MAX - T_MIN) / 2 + T_MIN;
  x(1) = (x(1) + 1) * (T_MAX - T_MIN) / 2 + T_MIN;
  x(2) = round((x(2) + 1) * (N_MAX - N_MIN) / 2 + N_MIN);
  x(3) = round((x(3) + 1) * (A_MAX - A_MIN) / 2 + A_MIN);
  return x;
}

void play_midi(RtMidiOut *midiout, MidiFile midi) {

  if (!midi.status()) {
    cerr << "[!] train: Unable to process MIDI data from arguments" << endl;
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

MidiFile sample(model m, VectorXd *seq, int seqsize) {

  MidiFile midi;
  vector<uint8_t> note{NOTE_ON, 0, 0};
  VectorXd x(IO_SIZE), n(IO_SIZE);
  double atime = 0;

  midi.absoluteTicks();
  midi.setTPQ(TPQ);
  midi.addTrack(1);

  if (seqsize) {

    for (size_t i = 0; i < (size_t)seqsize; i++) {

      note[1] = seq[i](2), note[2] = seq[i](3);
      midi.addEvent(1, (int)((atime + seq[i](0)) * TPQ * BPM / 60), note);

      note[2] = 0;
      midi.addEvent(1, (int)((atime + seq[i](0) + seq[i](1)) * TPQ * BPM / 60),
                    note);

      cout << "[*] generate: Recorded pre-entered note " << (int)seq[i](2)
           << " of attack " << (int)seq[i](3) << " ON to " << atime + seq[i](0)
           << " and OFF to " << atime + seq[i](0) + seq[i](1) << endl;

      atime += seq[i](0);
    }

    x = seq[seqsize - 1];

  } else {

    x = decode(VectorXd::Random(4));

    note[1] = x(2), note[2] = x(3);
    midi.addEvent(1, (int)((atime + x(0)) * TPQ * BPM / 60), note);

    note[2] = 0;
    midi.addEvent(1, (int)((atime + x(0) + x(1)) * TPQ * BPM / 60), note);

    cout << "[*] generate: Recorded pre-entered note " << (int)x(2)
         << " of attack " << (int)x(3) << " ON to " << atime + x(0)
         << " and OFF to " << atime + x(0) + x(1) << endl;

    atime += x(0);
  }

  for (size_t i = 0; i < SAMPLE_SIZE; i++) {

    x = predict(&m, x);
    n = decode(map(x, cap));

    note[1] = n(2), note[2] = n(3);
    midi.addEvent(1, (int)((atime + n(0)) * TPQ * BPM / 60), note);

    note[2] = 0;
    midi.addEvent(1, (int)((atime + n(0) + n(1)) * TPQ * BPM / 60), note);

    cout << "[*] generate: Recorded generated note " << (int)n(2)
         << " of attack " << (int)n(3) << " ON to " << atime + n(0)
         << " and OFF to " << atime + n(0) + n(1) << endl;

    atime += n(0);
  }

  midi.sortTracks();
  return midi;
}

model grad_desc(model m, MidiFile midi, double *loss) {

  model grad;
  mset_zero(&grad);
  int check0112 = 1;

  if (!midi.status()) {
    cerr << "[!] Unable to parse MIDI data from arguments, skipping batch..."
         << endl;
    return grad;
  }
  midi.absoluteTicks();
  midi.joinTracks();
  midi.doTimeAnalysis();
  midi.linkNotePairs();
  size_t batch_size = 0;
  VectorXd x[MAX_BATCH_SIZE], y[MAX_BATCH_SIZE], a[MAX_BATCH_SIZE][LAYER_NUM],
      d;
  double ptime = 0;

  *loss = 0;

  for (size_t i = 0; i < (size_t)midi[0].size(); i++) {

    if (midi[0][i].isPatchChange())
      check0112 = 79 < midi[0][i][1] && midi[0][i][1] < 128;

    if (midi[0][i].isNoteOn() && !check0112) {

      batch_size++;

      for (size_t j = 0; j < LAYER_NUM; j++)
        a[batch_size - 1][j] = m.a[j];
      x[batch_size - 1].resize(IO_SIZE);
      x[batch_size - 1] << midi[0][i].seconds - ptime,
          midi[0][i].getDurationInSeconds(), midi[0][i][1], midi[0][i][2];
      x[batch_size - 1] = encode(x[batch_size - 1]);

      y[batch_size - 1] = predict(&m, x[batch_size - 1]);
      if (batch_size > 1)
        for (size_t j = 0; j < IO_SIZE; j++)
          *loss += (x[batch_size - 1](j) - y[batch_size - 2](j)) *
                   (x[batch_size - 1](j) - y[batch_size - 2](j)) / IO_SIZE;

      ptime = midi[0][i].seconds;
    }
  }

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
          grad.a[j - 1](k) *= 1 - a[i][j - 1](k) * a[i][j - 1](k);
        grad.waa[j - 1] = grad.a[j - 1] * a[i - 1][j - 1].transpose();
        grad.a[j - 1] = m.waa[j - 1].transpose() * grad.a[j - 1];
      }

      if (i > 1) {
        grad.wxy[j] += d * a[i - 1][j - 1].transpose();
        grad.b[j] += d;
        grad.a[j - 1] += m.wxy[j].transpose() * d;
      }

      d = grad.a[j - 1];
      for (size_t k = 0; k < H_SIZE; k++)
        d(k) *= 1 - a[i - 1][j - 1](k) * a[i - 1][j - 1](k);
    }

    grad.wxy[0] += d * x[i - 1].transpose();
    grad.b[0] += d;
  }

  return *mop_div(&grad, (double)batch_size);
}

void train(model *m, MidiFile *midi, size_t batch_num, size_t sepoch,
           RtMidiOut *midiout) {

  model grad;
  mset_zero(&grad);
  double loss = 0, batch_loss = 0;
  MidiFile sample_midi;

  for (size_t epoch = sepoch; epoch < EPOCH_NUM; epoch++) {

    if (epoch != 0 && epoch % EPOCH_SAMPLE_PERIOD == 0) {

      cout << "[*] Saving model for epoch " << epoch << "..." << endl;
      msv_save(m, MODEL_DIR + "model-" + to_string(epoch) + ".json");
      cout << "[*] Model successfully saved" << endl;

      cout << "[*] Generating MIDI sample for epoch " << epoch << "..." << endl;
      sample_midi = sample(*m, {}, 0);
      cout << "[*] MIDI sample successfully generated" << endl;

      sample_midi.write(SAMPLE_DIR + "sample-" + to_string(epoch) + ".mid");
      cout << "[*] MIDI sample saved to " << SAMPLE_DIR << "sample-" << epoch
           << ".mid" << endl;

      if (midiout != NULL) {
        thread(play_midi, midiout, sample_midi).detach();
        cout << "[*] Now playing: MIDI sample for epoch " << epoch << endl;
      }
    }

    for (size_t batch = 0; batch < batch_num; batch++) {

      if (batch != 0 && batch % MINI_BATCH_SIZE == 0) {

        mop_add(m, *mop_mult(mop_div(&grad, MINI_BATCH_SIZE), -LEARNING_RATE));

        cout << "[*] Model updated over " << MINI_BATCH_SIZE
             << " mini-batches on batch " << batch << ", with loss "
             << loss / MINI_BATCH_SIZE << endl;

        loss = 0;
      }

      if (!midi[batch].status()) {
        cerr << "[!] train: Unable to parse MIDI data from arguments, "
                "skipping: batch "
             << batch + 1 << endl;
        continue;
      }

      mop_add(&grad, grad_desc(*m, midi[batch], &batch_loss));
      loss += batch_loss;
    }

    mop_add(m, *mop_mult(mop_div(&grad, ((batch_num % MINI_BATCH_SIZE)
                                             ? batch_num % MINI_BATCH_SIZE
                                             : MINI_BATCH_SIZE)),
                         -LEARNING_RATE));
    cout << "[*] Model updated over "
         << ((batch_num % MINI_BATCH_SIZE) ? batch_num % MINI_BATCH_SIZE
                                           : MINI_BATCH_SIZE)
         << " mini-batches on batch " << batch_num << ", with loss "
         << loss / ((batch_num % MINI_BATCH_SIZE) ? batch_num % MINI_BATCH_SIZE
                                                  : MINI_BATCH_SIZE)
         << ::endl;

    loss = 0;

    cout << "[*] Epoch " << epoch + 1 << "/" << EPOCH_NUM << endl;
  }

  cout << "[*] Saving model for epoch " << EPOCH_NUM << "..." << endl;
  msv_save(m, MODEL_DIR + "model-" + to_string(EPOCH_NUM) + ".json");
  cout << "[*] Model successfully saved" << endl;

  cout << "[*] Generating MIDI sample for epoch " << EPOCH_NUM << "..." << endl;
  sample_midi = sample(*m, {}, 0);
  cout << "[*] MIDI sample successfully generated" << endl;

  sample_midi.write(SAMPLE_DIR + "sample-" + to_string(EPOCH_NUM) + ".mid");
  cout << "[*] MIDI sample saved to " << SAMPLE_DIR << "sample-" << EPOCH_NUM
       << ".mid" << endl;

  if (midiout != NULL) {
    thread(play_midi, midiout, sample_midi).join();
    cout << "[*] Now playing: MIDI sample for epoch " << EPOCH_NUM << "..."
         << endl;
  }
}
