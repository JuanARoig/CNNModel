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
        //cnnLayerAmount includes Convolution, reLu, and pooling.
        //if cnn never reaches transition and runs out of kernels,
        //user should also be notified of this occurence.
        public double[] GeneralCNN(double[,,] inputData, int cnnLayerAmount) 
        {
            int elementAmount = inputData.GetLength(0)*inputData.GetLength(1)*inputData.GetLength(2); 
            while(elementAmount < this.transitionElementAmount)
            {
                //reassign elementAmount after convolution and pooling
            }
            //gonna use two layers for fully connected portion.
        }
    }
}