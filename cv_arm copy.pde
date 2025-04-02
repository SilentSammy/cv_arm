PShape base, shoulder, upArm, loArm, end;
float rotX, rotY;
float posX=0, posY=-60, posZ=0;
float alpha, beta, gamma;
float F = 50;
float T = 70;
float millisOld, gTime, gSpeed = 4;

double[] end_eff_image = new double[] {0.38, 0.26};
double[] normalize(double x, double y) {
    double normX = x / 0.38;
    double normY = y / 0.26;
    return new double[] { normX, normY };
}
public static double[] denormalize(double x, double y) {
    double denormX = x * -60;
    double denormY = y * -40 + 51;
    return new double[] { denormX, denormY };
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

ArrayList<PVector> readNormalizedPoints(String filename) {
  ArrayList<PVector> points = new ArrayList<PVector>();
  String[] lines = loadStrings(filename);

  for (String line : lines) {
    line = line.trim();
    if (line.startsWith("(") && line.endsWith(")")) {
      line = line.substring(1, line.length() - 1); // Remove parentheses
      String[] parts = line.split(",");
      if (parts.length == 2) {
        float x = float(trim(parts[0]));
        float y = float(trim(parts[1]));
        points.add(new PVector(x, y));
      }
    }
  }

  return points;
}

void writePos(){
  IK();
  setTime();
  ArrayList<PVector> points = readNormalizedPoints("coords_normalized.txt");
  PVector end_eff_image = points.get(0);  // Correct way to get first element
  PVector base_image = points.get(1);  // Correct way to get second element

  end_eff_image = new PVector(abs(end_eff_image.x - base_image.x), abs(end_eff_image.y - base_image.y));
  

  double[] norm_end_eff = normalize(end_eff_image.x, end_eff_image.y);
  double[] denorm_end_eff = denormalize(norm_end_eff[0], norm_end_eff[1]);

  posY = (float)denorm_end_eff[0];
  posZ = (float)denorm_end_eff[1];
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

    drawReferenceFrame();

    for (int i=0; i < Xsphere.length; i++) {
     pushMatrix();
     translate(-Ysphere[i], -Zsphere[i]-11, -Xsphere[i]);
     fill (#D003FF, 25);
     sphere (5);
     popMatrix();
    }

    fill(#FFE308);  
    translate(0,-40,0);   
    drawReferenceFrame();
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
