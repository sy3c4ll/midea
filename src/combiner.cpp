#include <cstdio>
#include <midifile/MidiFile.h>
using smf::MidiFile;

int main(int argc, char **argv) {
  MidiFile midi(argv[1]);
  size_t count = 0;
  midi.joinTracks();
  midi.write(argv[1]);
  for (size_t i = 0; i < midi[0].getSize(); i++)
    if (midi[0][i].isNoteOn())
      count++;
  printf("%d notes\n", (int)count);
  return 0;
}
