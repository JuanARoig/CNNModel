using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace neuralnet
{
    partial class ConvolutionalNN
    {
        //use partialderivs to implement gradient descent in backpropagation
        public void CrossEntropyTraining(List<double[,,] inputDataList, List<double[]> outputDataList)
        {
            foreach(double[,,] inputData, double[] expectedOutput in inputDataList, outputDataList)
            {
                foreach()
                {
                    double original = this.CrossEntropyLoss(inputData, expectedOutput);
                    
                }
            }
        }
        public void MeanSquaredTraining(List<double[]> inputDataList, List<double[]> outputDataList)
        {
            foreach(double[,,] inputData, double[] expectedOutput in inputDataList, outputDataList)
            {
                foreach()
                {
                    double original = this.MeanSquaredCost(inputData, expectedOutput);
                }
            }
        }
    }
}