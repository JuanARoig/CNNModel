using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace neuralnet{

	partial class ConvolutionalNN
	{
		public ConvolutionalNN ConvolutionalNN() { }
		public ConvolutionalNN ConvolutionalNN(int[,] kernelSizes, int transitionElementAmount)//kernelSizes j-hat sizes should be of size 3.
		{
			this.kernelSizes = kernelSizes;
			this.transitionElementAmount = transitionElementAmount;
			//when dimensions fall below the transitionDimensions, network will transition from
			//convolutional to fully connected.
		}	
	}
}