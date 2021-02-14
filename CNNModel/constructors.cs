using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace neuralnet{

	partial class ConvolutionalNN
	{
		public ConvolutionalNN ConvolutionalNN() { }
		public ConvolutionalNN ConvolutionalNN(int[,] kernelSizes, int[3] transitionDimensions)//kernelSizes j-hat sizes should be of size 3.
		{
			this.kernelSizes = kernelSizes;
			this.transitionDimensions = transitionDimensions;
			//when dimensions fall below the transitionDimensions, network will transition from
			//convolutional to fully connected.
		}	
	}
}