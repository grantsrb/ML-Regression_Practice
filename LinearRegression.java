import java.lang.Math;
import java.util.Random;

public class LinearRegression {
  public static void main(String[] args) {
    // Goal is to create a linear regression function for a given dataset. We will attempt this process with both a pure dataset and a randomized one.

    // Create datasets
    Random randGenerator = new Random();
    double noise;
    double noiseModifier = 1;
    double[] pureData = new double[5];
    double[] randomData = new double[5];
    for (int i = 0; i < 5; i++) {
      pureData[i] = (double) i + 1.0d;
      noise = randGenerator.nextDouble() * noiseModifier;
      randomData[i] = (double) i + 1 + (randGenerator.nextBoolean() ? noise:-noise);
    }

    // Create randomized hypothesis parameters
    double theta0 = 1.0d;
    double theta1 = 2.0d;
    double temp0, temp1;
    double learningRate = 0.0001d;
    double tolerance = 0.00000001d;
    int count0 = 0, count1 = 0;

    // Take partial derivatives of Cost Fxn and minimize function costFxn(theta0, theta1, randomData)>tolerance
    while (count0 < 1000 || count1 < 1000) {
      // Evaluate minimization of parameters
      temp0 = theta0 - learningRate*partialFxn0(theta0, theta1, randomData);
      temp1 = theta1 - learningRate*partialFxn1(theta0, theta1, randomData);
      if(Math.pow((theta0 - temp0),2) < tolerance) {
        count0++;
      }
      if(Math.abs((theta1 - temp1)) < tolerance) {
        count1++;
      }
      // Update parameters
      theta0 = temp0;
      theta1 = temp1;
    }

    System.out.println("Parameters:\ntheta0: " + theta0);
    System.out.println("theta1: " + theta1 + "\n");

    for(int i = 0; i < 5; i++) {
      System.out.println("Data (" + i + "): " + randomData[i]);
      System.out.println("Fitted point: " + hypothesis(theta0, theta1, i+1.0d) + "\n");
    }

  }

  // Create Hypothesis function
  public static double hypothesis(double theta0, double theta1, double x) {
    return theta0 + theta1*x;
  }

  // Create cost function J(ø0, ø1) = ∑(squareDiffs)
  public static double costFxn(double theta0, double theta1, double[] data) {
    double diffSum = 0;
    for(int i = 0; i < data.length; i++) {
      diffSum += Math.pow((data[i] - hypothesis(theta0, theta1, i+1)), 2);
    }
    return diffSum * (1.0d/(2.0d*data.length));
  }

  // Partial derivative of cost function with respect to theta0
  public static double partialFxn0(double theta0, double theta1, double[] data) {
    double diffSum = 0;
    for(int i = 0; i < data.length; i++) {
      diffSum += 2*(hypothesis(theta0, theta1, i+1) - data[i]);
    }
    return diffSum * (1.0d/(2.0d*data.length));
  }

  // Partial derivative of cost function with respect to theta1
  public static double partialFxn1(double theta0, double theta1, double[] data) {
    double diffSum = 0;
    for(int i = 0; i < data.length; i++) {
      diffSum += 2*(i+1)*(hypothesis(theta0, theta1, i+1) - data[i]);
    }
    return diffSum * (1.0d/(2.0d*data.length));
  }
}
