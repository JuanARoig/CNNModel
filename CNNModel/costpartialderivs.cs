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
            double[,,] targetKernel = this.convLayerKernels[listpos];

            targetKernel[arrpos1, arrpos2, arrpos3] += r;
            double affected = this.CrossEntropyLoss(inputData, expectedOutput);
            targetKernel[arrpos1, arrpos2, arrpos3] -= r;
            double final = (affected - original)/r;
            return final;
        }
        private double CEDerivWeights(double[,,] inputData, double[] expectedOutput, int listpos, int arrpos1, int arrpos2) 
        { 
            double r = 0.00000001;
            double original = this.CrossEntropyLoss(inputData, expectedOutput);
            double[,] targetWeights = this.weights[listpos];

            targetWeights[arrpos1, arrpos2] += r;
            double affected = this.CrossEntropyLoss(inputData, expectedOutput);
            targetWeights[arrpos1, arrpos2] -= r;
            double final = (affected - original)/r;
            return final;
        }
        private double CEDerivBiases(double[,,] inputData, double[] expectedOutput, int listpos, int arrpos) 
        { 
            double r = 0.00000001;
            double original = this.CrossEntropyLoss(inputData, expectedOutput);
            double[] targetBiases = this.biases[listpos];

            targetBiases[arrpos] += r;
            double affected = this.CrossEntropyLoss(inputData, expectedOutput);
            targetBiases[arrpos] -= r;
            double final = (affected - original)/r;
            return final;
        }

        private double MSEDerivKernels(double[,,] inputData, double[] expectedOutput, int listpos, int arrpos1, int arrpos2, int arrpos3) 
        { 
            double r = 0.00000001;
            double original = this.MeanSquaredCost(inputData, expectedOutput);
            double[,,] targetKernel = this.convLayerKernels[listpos];

            targetKernel[arrpos1, arrpos2, arrpos3] += r;
            double affected = this.MeanSquaredCost(inputData, expectedOutput);
            targetKernel[arrpos1, arrpos2, arrpos3] -= r;
            double final = (affected - original)/r;
            return final;
        }
        private double MSEDerivWeights(double[,,] inputData, double[] expectedOutput, int listpos, int arrpos1, int arrpos2) 
        { 
            double r = 0.00000001;
            double original = this.MeanSquaredCost(inputData, expectedOutput);
            double[,] targetWeights = this.weights[listpos];

            targetWeights[arrpos1, arrpos2] += r;
            double affected = this.MeanSquaredCost(inputData, expectedOutput);
            targetWeights[arrpos1, arrpos2] -= r;
            double final = (affected - original)/r;
            return final;
        }
        private double MSEDerivBiases(double[,,] inputData, double[] expectedOutput, int listpos, int arrpos) 
        { 
            double r = 0.00000001;
            double original = this.MeanSquaredCost(inputData, expectedOutput);
            double[] targetBiases = this.biases[listpos];

            targetBiases[arrpos] += r;
            double affected = this.MeanSquaredCost(inputData, expectedOutput);
            targetBiases -= r;
            double final = (affected - original)/r;
            return final;
        }
    }
}