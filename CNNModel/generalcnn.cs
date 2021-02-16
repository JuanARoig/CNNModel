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
            while(elementAmount < this.transitionElementAmount && kernelListCounter < convLayerKernels.Capacity)
            {
                //reassign elementAmount after convolution and pooling
                int kernelIndex = 0;
                int convLayerIndex = 0;
                this.convLayerNetwork.Add(inputData);
                while ((current.GetLength(0)*current.GetLength(1)*current.GetLength(2) < transitionElementAmount) 
                && kernelIndex < convLayerKernels.Capacity)
                {
                    double[,,] currentKernel = convLayerKernels[kernelIndex];
                    double[,,] current = convLayerNetwork[convLayerIndex];
                    double[] currentKernelDimensions = new double[] { currentKernel.GetLength(0), 
                                                                      currentKernel.GetLength(1),
                                                                      currentKernel.GetLength(2) }
                    this.convLayerNetwork.Add(MaxPoolingOperation(ConvolutionOperation(current, currentKernel),
                                                                                       currentKernelDimensions));
                }
            }
            //FCLayerSizes is assigned values.
            Random random = new Random(this.seed);//used for weights and biases once FCNN specs are known.
            //gonna use two layers for fully connected portion.
        }
    }
}