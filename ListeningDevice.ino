#include <TMRpcm.h>
#include <SD.h>
#include <SPI.h>

#define SD_ChipSelectPin 10

TMRpcm audio;
unsigned long i = 0;
int soundFile = 0;
bool recStatus= 0;

void setup() {

  pinMode(A0, INPUT);
  pinMode(2, INPUT_PULLUP);
  pinMode(6, OUTPUT);
  attachInterrupt(0, button, LOW);
  SD.begin(SD_ChipSelectPin);
  audio.CSPin = SD_ChipSelectPin;

}

void loop() {
}

void button() {
  while (i < 300000) {
    i++;
  }
  i = 0;
  if (recStatus == 0) {
    recStatus = 1;
    audiofile++;
    digitalWrite(6, HIGH);
    switch (soundFile) {
      case 1: audio.startRecording("1.wav", 44100, A0); break;
      case 2: audio.startRecording("2.wav", 44100, A0); break;
      case 3: audio.startRecording("3.wav", 44100, A0); break;

    }
  }
  else {
    recStatus = 0;
    digitalWrite(6, LOW);
    switch (soundFile) {
      case 1: audio.stopRecording("1.wav"); break;
      case 2: audio.stopRecording("2.wav"); break;
      case 3: audio.stopRecording("3.wav"); break;

    }
  }
}
