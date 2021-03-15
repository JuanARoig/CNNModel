using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace neuralnet
{
    public partial class ConvolutionalNN
    {
        private double CEPDerivKernels(double original, double[,,] inputData, double[] expectedOutput, int listpos, int arrpos1, int arrpos2, int arrpos3) 
        { 
            double r = 0.00000001;
            double[,,] targetKernel = this.convLayerKernels[listpos];

            targetKernel[arrpos1, arrpos2, arrpos3] += r;
            double affected = this.CrossEntropyLoss(inputData, expectedOutput);
            targetKernel[arrpos1, arrpos2, arrpos3] -= r;
            double final = (affected - original)/r;
            return final;
        }
        private double CEPDerivWeights(double original, double[,,] inputData, double[] expectedOutput, int listpos, int arrpos1, int arrpos2) 
        { 
            double r = 0.00000001;
            double[,] targetWeights = this.weights[listpos];

            targetWeights[arrpos1, arrpos2] += r;
            double affected = this.CrossEntropyLoss(inputData, expectedOutput);
            targetWeights[arrpos1, arrpos2] -= r;
            double final = (affected - original)/r;
            return final;
        }
        private double CEPDerivBiases(double original, double[,,] inputData, double[] expectedOutput, int listpos, int arrpos) 
        { 
            double r = 0.00000001;
            double[] targetBiases = this.biases[listpos];

            targetBiases[arrpos] += r;
            double affected = this.CrossEntropyLoss(inputData, expectedOutput);
            targetBiases[arrpos] -= r;
            double final = (affected - original)/r;
            return final;
        }

        private double MSEPDerivKernels(double original, double[,,] inputData, double[] expectedOutput, int listpos, int arrpos1, int arrpos2, int arrpos3) 
        { 
            double r = 0.00000001;
            double[,,] targetKernel = this.convLayerKernels[listpos];

            targetKernel[arrpos1, arrpos2, arrpos3] += r;
            double affected = this.MeanSquaredCost(inputData, expectedOutput);
            targetKernel[arrpos1, arrpos2, arrpos3] -= r;
            double final = (affected - original)/r;
            return final;
        }
        private double MSEPDerivWeights(double original, double[,,] inputData, double[] expectedOutput, int listpos, int arrpos1, int arrpos2) 
        { 
            double r = 0.00000001;
            double[,] targetWeights = this.weights[listpos];

            targetWeights[arrpos1, arrpos2] += r;
            double affected = this.MeanSquaredCost(inputData, expectedOutput);
            targetWeights[arrpos1, arrpos2] -= r;
            double final = (affected - original)/r;
            return final;
        }
        private double MSEPDerivBiases(double original, double[,,] inputData, double[] expectedOutput, int listpos, int arrpos) 
        { 
            double r = 0.00000001;
            double[] targetBiases = this.biases[listpos];

            targetBiases[arrpos] += r;
            double affected = this.MeanSquaredCost(inputData, expectedOutput);
            targetBiases[arrpos] -= r;
            double final = (affected - original)/r;
            return final;
        }
    }
}