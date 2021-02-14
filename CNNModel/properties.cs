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

        private List<double[]> biases;
        private List<double[,]> weights;

        private List<double[]> biasupdates;
        private List<double[,]> weightsupdates;

        //Convolutional Neural Net Properties
        private int transitionElementAmount;//Dimensions boundary in order to transition to FCNN.
        private int[,] kernelSizes;//kernel1Size is the cubic dimensions of the kernel. j-hat should be of size 3.
	    private List<double[,,]> convLayerNetwork;//may not be necessary
	    private List<double[,,]> convLayerKernels;
	    private List<double[,,]> convLayerKernelUpdates;
    }
}