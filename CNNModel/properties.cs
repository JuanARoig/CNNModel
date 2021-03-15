using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace neuralnet
{
    public partial class ConvolutionalNN
    {
        //Fully Connected Neural Net Classifier Properties
        public int seed { get; set; }
	    public int FCLayerAmount { get; set; }
        private int finalOutputSize;
	    private int[] FCLayerSizes;//instantiated after cnnlayers completed.

        private List<double[]> biases;
        private List<double[,]> weights;

        private List<double[]> biasupdates;
        private List<double[,]> weightsupdates;
        //Convolutional Neural Net Properties
        private int transitionElementAmount;//Dimensions boundary in order to transition to FCNN.
        private int[,] kernelSizes;//kernel1Size is the cubic dimensions of the kernel. j-hat should be of size 3.
	    private List<double[,,]> convLayerNetwork;
	    private List<double[,,]> convLayerKernels;
	    private List<double[,,]> convLayerKernelUpdates;
    }
}