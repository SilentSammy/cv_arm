PShape base, shoulder, upArm, loArm, end;
float rotX, rotY;
float posX=0, posY=-60, posZ=0;
float alpha, beta, gamma;
float F = 50;
float T = 70;
float millisOld, gTime, gSpeed = 4;

void read_end_eff_pos() {
  // Attempt to load file lines
  String[] fileLines = loadStrings("end_eff.txt");
  if (fileLines == null || fileLines.length < 3) {
    println("Warning: File unreadable or not enough lines (need at least 3).");
    return;
  }
  
  ArrayList<Float> validFloats = new ArrayList<Float>();
  for (String line : fileLines) {
    try {
      float value = Float.parseFloat(line.trim());
      validFloats.add(value);
    } catch (NumberFormatException e) {
      println("Warning: Invalid float value encountered: " + line);
      // Continue to next line
    }
  }
  
  if (validFloats.size() < 3) {
    println("Warning: Not enough valid float values found.");
    return;
  }
  
  // Only assign if we have at least 3 valid floats.
  posX = validFloats.get(0);
  posY = validFloats.get(1);
  posZ = validFloats.get(2);
}

void IK(){
  float X = posX;
  float Y = posY;
  float Z = posZ;

  float L = sqrt(Y*Y+X*X);
  float dia = sqrt(Z*Z+L*L);

  alpha = PI/2-(atan2(L, Z)+acos((T*T-F*F-dia*dia)/(-2*F*dia)));
  beta = -PI+acos((dia*dia-T*T-F*F)/(-2*F*T));
  gamma = atan2(Y, X);
}
void setTime(){
  gTime += ((float)millis()/1000 - millisOld)*(gSpeed/4);
  if(gTime >= 4)  gTime = 0;  
  millisOld = (float)millis()/1000;
}

void writePos(){
  IK();
  setTime();
  read_end_eff_pos();
}

void drawReferenceFrame() {
  stroke(255, 0, 0); // X-axis in red
  line(0, 0, 0, 100, 0, 0);
  
  stroke(0, 255, 0); // Y-axis in green
  line(0, 0, 0, 0, 100, 0);
  
  stroke(0, 0, 255); // Z-axis in blue
  line(0, 0, 0, 0, 0, -100);
}

float[] Xsphere = new float[1];
float[] Ysphere = new float[1];
float[] Zsphere = new float[1];

void setup(){
    size(1200, 800, OPENGL);
    surface.setResizable(true);
    
    base = loadShape("r5.obj");
    shoulder = loadShape("r1.obj");
    upArm = loadShape("r2.obj");
    loArm = loadShape("r3.obj");
    end = loadShape("r4.obj");
    
    shoulder.disableStyle();
    upArm.disableStyle();
    loArm.disableStyle(); 
}

void draw(){
   writePos();
   background(32);
   smooth();
   lights(); 
   directionalLight(51, 102, 126, -1, 0, 0);
    
    for (int i=0; i< Xsphere.length - 1; i++) {
    Xsphere[i] = Xsphere[i + 1];
    Ysphere[i] = Ysphere[i + 1];
    Zsphere[i] = Zsphere[i + 1];
    }
    
    Xsphere[Xsphere.length - 1] = posX;
    Ysphere[Ysphere.length - 1] = posY;
    Zsphere[Zsphere.length - 1] = posZ;
   
   noStroke();
   
    translate(width/2,height/2);
    rotateX(rotX);
    rotateY(-rotY);
    scale(-4);

    // drawReferenceFrame();

    for (int i=0; i < Xsphere.length; i++) {
     pushMatrix();
     translate(-Ysphere[i], -Zsphere[i]-11, -Xsphere[i]);
     stroke(0, 0, 255);
     sphere (5);
     popMatrix();
    }

    fill(#FFE308);  
    translate(0,-40,0);   
    // drawReferenceFrame();
    shape(base);

    fill(#FFE308);
    translate(0, 4, 0);
    rotateY(gamma);
    shape(shoulder);

    translate(0, 25, 0);
    rotateY(PI);
    rotateX(alpha);
    shape(upArm);

    translate(0, 0, 50);
    rotateY(PI);
    rotateX(beta);
    shape(loArm);

    translate(0, 0, -50);
    rotateY(PI);
    shape(end);
}

void mouseDragged(){
    rotY -= (mouseX - pmouseX) * 0.01;
    rotX -= (mouseY - pmouseY) * 0.01;
}
