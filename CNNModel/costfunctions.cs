using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace neuralnet
{
    partial class ConvolutionalNN
    {
        //mean squared error function
        private double MeanSquaredCost(double[,,] inputData, double[] expectedOutput)
        {
            double finalError = 0;
            double[] output = GeneralCNN(inputData);
            for (int i = 0; i < output.Length; i++)
            {
                finalError += Math.Pow(output[i] - expectedOutput[i], 2);
            }
            finalError *= (1/output.Length);
            return finalError;
        }
        //cross entropy loss function
        private double CrossEntropyLoss(double[,,] inputData, double[] expectedOutput)
        {
            double finalError = 0;
            double[] output = GeneralCNN(inputData);
            for (int i = 0; i < output.Length; i++)
            {
                finalError += expectedOutput[i]*(Math.Log10(output[i]));
            }
            finalError *= (1/output.Length);
            return finalError;
        }
    }
}