using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace neuralnet
{
    partial class ConvolutionalNN
    {
        //use partialderivs to implement gradient descent in backpropagation
        public void CrossEntropyTraining(double learningRate, List<double[,,]> inputDataList, List<double[]> outputDataList)
        {
            foreach(double[,,] inputData, double[] expectedOutput in inputDataList, outputDataList)
            {
                
                double original = this.CrossEntropyLoss(inputData, expectedOutput);
                for (int x = 0; x < this.convLayerKernels.Capacity; x++)
                {
                    
                }
            }
        }
        public void MeanSquaredTraining(List<double[]> inputDataList, List<double[]> outputDataList)
        {
            foreach(double[,,] inputData, double[] expectedOutput in inputDataList, outputDataList)
            {
            
                double original = this.MeanSquaredCost(inputData, expectedOutput);
                for (int x = 0; x < this.convLayerKernels.Capacity; x++)
                {
                    
                }
            }
        }
    }
}