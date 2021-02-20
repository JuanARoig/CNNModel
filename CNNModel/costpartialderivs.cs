using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace neuralnet
{
    partial class ConvolutionalNN
    {
        private double CEPDerivKernels(double[,,] inputData, double[] expectedOutput, int listpos, int arrpos1, int arrpos2, int arrpos3) 
        { 
            double r = 0.00000001;
            double original = this.CrossEntropyLoss(inputData, expectedOutput);
        }
        private double CEDerivWeights(double[,,] inputData, double[] expectedOutput, int listpos, int arrpos1, int arrpos2) 
        { 
            double r = 0.00000001;
            double original = this.CrossEntropyLoss(inputData, expectedOutput);
        }
        private double CEDerivBiases(double[,,] inputData, double[] expectedOutput, int listpos, int arrpos) 
        { 
            double r = 0.00000001;
            double original = this.CrossEntropyLoss(inputData, expectedOutput);
        }

        private double MSEDerivKernels(double[,,] inputData, double[] expectedOutput, int listpos, int arrpos1, int arrpos2, int arrpos3) 
        { 
            double r = 0.00000001;
            double original = this.MeanSquaredCost(inputData, expectedOutput);
        }
        private double MSEDerivWeights(double[,,] inputData, double[] expectedOutput, int listpos, int arrpos1, int arrpos2) 
        { 
            double r = 0.00000001;
            double original = this.MeanSquaredCost(inputData, expectedOutput);
        }
        private double MSEDerivBiases(double[,,] inputData, double[] expectedOutput, int listpos, int arrpos) 
        { 
            double r = 0.00000001;
            double original = this.MeanSquaredCost(inputData, expectedOutput);
        }
    }
}