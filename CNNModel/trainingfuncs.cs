using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace neuralnet
{
    partial class ConvolutionalNN
    {
        //GD = Gradient Descent. Stochastic GD and Adam may be written later.
        public void CrossEntropyGDTraining(int reps, double learningRate, double lrMult, List<double[,,]> inputDataList, List<double[]> outputDataList)
        {
            for (int l = 0; l < reps; l++)
            {
                foreach(double[,,] inputData, double[] expectedOutput in inputDataList, outputDataList)
                {
                    double original = this.CrossEntropyLoss(inputData, expectedOutput);
                    for (int x = 0; x < this.convLayerKernels.Capacity; x++)
                    {
                        double[,,] currentKernel = this.convLayerKernels[x];
                        double[,,] kernelUpdateTensor = convLayerKernelUpdates[x];
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
                        double[,] weightsUpdateMatrix = this.weightsupdates[x];
                        for (int i = 0; i < currentWeights.GetLength(0); i++)
                        {
                            for (int j = 0; j < currentWeights.GetLength(1); j++)
                            {
                                  weightsUpdateMatrix[i, j] = (-1*learningRate*this.CEPDerivWeights(original, inputData, expectedOutput, x, i, j));                                 
                            }
                        }
                    }
                    for (int x = 0; x < this.biases.Capacity; x++)
                    {
                        double[] currentBiases = this.biases[x];
                        double[] biasUpdateVector = this.biasupdates[x];
                        for (int i = 0; i < currentBiases.Length; i++)
                        {
                            biasUpdateVector[i] = (-1*learningRate*this.CEPDerivBiases(original, inputData, expectedOutput, x, i));
                        }
                    }
                }
                learningRate *= lrMult;
                this.convLayerKernels = this.convLayerKernelUpdates;
                this.weights = this.weightsupdates;
                this.biases = this.biasupdates;
            }
        }
        public void MeanSquaredGDTraining(int reps, double learningRate, double lrMult, List<double[]> inputDataList, List<double[]> outputDataList)
        {
            for (int l = 0; l < reps; l++)
            {
                foreach(double[,,] inputData, double[] expectedOutput in inputDataList, outputDataList)
                {
                    double original = this.MeanSquaredCost(inputData, expectedOutput);
                    for (int x = 0; x < this.convLayerKernels.Capacity; x++)
                    {
                        double[,] currentKernel = this.convLayerKernels[x];
                        double[,,] kernelUpdateTensor = convLayerKernelUpdates[x];
                        for (int i = 0; i < currentKernel.GetLength(0); i++)
                        {
                            for (int j = 0; j < currentKernel.GetLength(1); j++)
                            {
                                for (int k = 0; k < currentKernel.GetLength(2); k++)
                                {
                                    kernelUpdateTensor[i, j, k] = (-1*learningRate*this.MSEPDerivKernels(original, inputData, expectedOutput, x, i, j, k));
                                }
                            }
                        }
                    }
                    for (int x = 0; x < this.weights.Capacity; x++)
                    {
                        double[,] currentWeights = this.weights[x];
                        double[,] weightsUpdateMatrix = this.weightsupdates[x];
                        for (int i = 0; i < currentWeights.GetLength(0); i++)
                        {
                            for (int j = 0; j < currentWeights.GetLength(1); j++)
                            {
                                weightsUpdateMatrix[i, j] = (-1*learningRate*this.MSEPDerivWeights(original, inputData, expectedOutput, x, i, j));
                            }
                        }
                    }
                    for (int x = 0; x < this.biases.Capacity; x++)
                    {
                        double[] currentBiases = this.biases[x];
                        double[] biasUpdateVector = this.biasupdates[x];
                        for (int i = 0; i < currentBiases.Length; i++)
                        {
                            biasUpdateVector[i] = (-1*learningRate*MSEPDerivBiases(original, inputData, expectedOutput, x, i));
                        }
                    }
                }
                learningRate *= lrMult;
                this.convLayerKernels = this.convLayerKernelUpdates;
                this.weights = this.weightsupdates;
                this.biases = this.biasupdates;
            }
        }
    }
}