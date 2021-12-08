#include <Servo.h>
#define LaserPin 9
Servo myservo;
unsigned int integerValue=0;
char incomingByte;

void setup ()
{
   Serial.begin(9600);
   pinMode (LaserPin, OUTPUT); 
   myservo.attach(8);
}

void loop () {
  while(Serial.available()){
     integerValue = 0;         // throw away previous integerValue
      while(1) {            // force into a loop until 'n' is received
        incomingByte = Serial.read();
        if (incomingByte == '\n') break;   // exit the while(1), we're done receiving
        if (incomingByte == -1) continue;  // if no characters are in the buffer read() returns -1
        integerValue *= 10;  // shift left 1 decimal place
        integerValue = ((incomingByte - 48) + integerValue);
      }
  }
  digitalWrite (LaserPin, HIGH);
  myservo.write(integerValue);

}
