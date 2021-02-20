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
            double finalError;
            double[] output = GeneralCNN(inputData);
            return finalError;
        }
        //cross entropy loss function
        private double CrossEntropyLoss(double[,,] inputData, double[] expectedOutput)
        {
            double finalError;
            double[] output = GeneralCNN(inputData);
            return finalError;
        }
    }
}