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
            for (int l = 0; l < 15; l++)
            {
                foreach(double[,,] inputData, double[] expectedOutput in inputDataList, outputDataList)
                {
                
                    double original = this.CrossEntropyLoss(inputData, expectedOutput);
                    for (int x = 0; x < this.convLayerKernels.Capacity; x++)
                    {
                        double[,,] currentKernel = this.convLayerKernels[x];
                        for (int i = 0; i < currentKernel.GetLength(0); i++)
                        {
                            for (int j = 0; j < currentKernel.GetLength(1); j++)
                            {
                                for (int k = 0; k < currentKernel.GetLength(2); k++)
                                {
                                
                                }
                            }
                        }
                    }
                    for (int x = 0; x < this.weights.Capacity; x++)
                    {
                        double[,] currentWeights = this.weights[x];
                        for (int i = 0; i < currentWeights.GetLength(0); i++)
                        {
                            for (int j = 0; j < currentWeights.GetLength(1); j++)
                            {

                            }
                        }
                    }
                    for (int x = 0; x < this.biases.Capacity; x++)
                    {
                        double[] currentBiases = this.biases[x];
                        for (int i = 0; i < currentBiases.Length; i++)
                        {

                        }
                    }
                }
            }
        }
        public void MeanSquaredTraining(List<double[]> inputDataList, List<double[]> outputDataList)
        {
            for (int l = 0; l < 15; l++)
            {
                foreach(double[,,] inputData, double[] expectedOutput in inputDataList, outputDataList)
                {
            
                    double original = this.MeanSquaredCost(inputData, expectedOutput);
                    for (int x = 0; x < this.convLayerKernels.Capacity; x++)
                    {
                        double[,] currentKernel = this.convLayerKernels[x];
                        for (int i = 0; i < currentKernel.GetLength(0); i++)
                        {
                            for (int j = 0; j < currentKernel.GetLength(1); j++)
                            {
                                for (int k = 0; k < currentKernel.GetLength(2); k++)
                                {
                                
                                }
                            }
                        }
                    }
                    for (int x = 0; x < this.weights.Capacity; x++)
                    {
                        double[,] currentWeights = this.weights[x];
                        for (int i = 0; i < currentWeights.GetLength(0); i++)
                        {
                            for (int j = 0; j < currentWeights.GetLength(1); j++)
                            {

                            }
                        }
                    }
                    for (int x = 0; x < this.biases.Capacity; x++)
                    {
                        double[] currentBiases = this.biases[x];
                        for (int i = 0; i < currentBiases.Length; i++)
                        {

                        }
                    }
                }
            }
        }
    }
}