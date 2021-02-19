using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace neuralnet
{
    partial class ConvolutionalNN
    {
        //User is allowed to enter amount of kernels to be used.
        //If the network is forced to transition before all kernels are used,
        //user will be notified.
        //cnn includes Convolution, reLu, and pooling.
        //if cnn never reaches transition and runs out of kernels,
        //user should also be notified of this occurence.
        private double[] GeneralCNN(double[,,] inputData, int cnnLayerAmount, int this.finalOutputSize) 
        {
            int elementAmount = inputData.GetLength(0)*inputData.GetLength(1)*inputData.GetLength(2);
            int kernelListCounter = 0; 
            double[,,] current;
            while(elementAmount < this.transitionElementAmount && kernelListCounter < convLayerKernels.Capacity)
            {
                //reassign elementAmount after convolution and pooling
                int kernelIndex = 0;
                int convLayerIndex = 0;
                this.convLayerNetwork.Add(inputData);
                current = convLayerNetwork[convLayerIndex];//first time
                while ((current.GetLength(0)*current.GetLength(1)*current.GetLength(2) < transitionElementAmount) 
                && kernelIndex < convLayerKernels.Capacity)
                {
                    double[,,] currentKernel = convLayerKernels[kernelIndex];
                    current = convLayerNetwork[convLayerIndex];
                    double[] currentKernelDimensions = new double[] { currentKernel.GetLength(0), 
                                                                      currentKernel.GetLength(1),
                                                                      currentKernel.GetLength(2) }
                    this.convLayerNetwork.Add(MaxPoolingOperation(ConvolutionOperation(current, currentKernel),
                                                                                       currentKernelDimensions));
                }
            }
            //converting 3d tensor to 1d vector
            double[] transitionVector = new double[current.GetLength(0)*current.GetLength(1)*current.GetLength(2)];
            double[,,] transitionTensor = current[current.Capacity - 1];
            int transitionAccum = 0;
            for (int i = 0; i < current[current.Capacity - 1].GetLength(0); i++)
            {
                for (int j = 0; j < current[current.Capacity - 1].GetLength(1); j++)
                {
                    for (int k = 0; k < current[current.Capacity - 1].GetLength(2); k++)
                    {
                        transitionVector[transitionAccum] = transitionTensor[i, j, k];
                        transitionAccum++;
                    }
                }
            }
            double[] currentLayer;
            for (int i = 0; i < this.weights.Capacity; i++)
            {
                if(i == 0)
                {
                    currentLayer = this.Add(MatrixVectorProduct(weights[i], transitionVector), biases[i]);
                }
                else
                {
                    currentLayer = this.Add(MatrixVectorProduct(weights[i], currentLayer), biases[i]);
                }
            }
            finalLayer = currentLayer;
            return finalLayer;
        }
    }
}