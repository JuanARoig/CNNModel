using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace neuralnet
{
    partial class ConvolutionalNN
    {
        //Fully Connected Neural Net Classifier Properties
	    private int FCLayerAmount;
	    private int[] FCLayerSizes;

        //Convolutional Neural Net Properties
        private int[,] kernelSizes;//kernel1Size is the cubic dimensions of the kernel. j-hat should be of size 3.
	    private List<double[,,]> convLayerNetwork;
	    private List<double[,,]> convLayerKernel;
	    private List<double[,,]> convLayerKernelUpdates;
    }
}