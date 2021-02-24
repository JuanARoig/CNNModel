using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace neuralnet
{
    partial class ConvolutionalNN
    {
        //use partialderivs to implement gradient descent in backpropagation
        public void CrossEntropyTraining(int reps, double learningRate, double lrMult, List<double[,,]> inputDataList, List<double[]> outputDataList)
        {
            for (int l = 0; l < reps; l++)
            {
                foreach(double[,,] inputData, double[] expectedOutput in inputDataList, outputDataList)
                {
                    double original = this.CrossEntropyLoss(inputData, expectedOutput);
                    for (int x = 0; x < this.convLayerKernels.Capacity; x++)
                    {
                        double[,,] currentKernel = this.convLayerKernels[x];
                        double[,,] kernelUpdateTensor = new double[currentKernel.GetLength(0),
                                                                   currentKernel.GetLength(1),
                                                                   currentKernel.GetLength(2)];
                        this.convLayerKernelUpdates[x] = kernelUpdateTensor;  
                        for (int i = 0; i < currentKernel.GetLength(0); i++)
                        {
                            for (int j = 0; j < currentKernel.GetLength(1); j++)
                            {
                                for (int k = 0; k < currentKernel.GetLength(2); k++)
                                {
                                    kernelUpdateTensor[i, j, k] = (-1*learningRate*this.CEPDerivKernels(original, inputData, expectedOutput, x, i, j, k));
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
                learningRate *= lrMult;
            }
        }
        public void MeanSquaredTraining(int reps, double learningRate, double lrMult, List<double[]> inputDataList, List<double[]> outputDataList)
        {
            for (int l = 0; l < reps; l++)
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
                learningRate *= lrMult;
            }
        }
    }
}