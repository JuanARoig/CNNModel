using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace neuralnet{

	partial class ConvolutionalNN
	{
		public ConvolutionalNN ConvolutionalNN() { }
		public ConvolutionalNN ConvolutionalNN(int[,] kernelSizes)//kernelSizes j-hat sizes should be of size 3.
		{
			this.kernelSizes = kernelSizes;
		}	
	}
}