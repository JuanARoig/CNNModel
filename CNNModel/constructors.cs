using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace neuralnet{

	partial class ConvolutionalNN
	{
		public ConvolutionalNN ConvolutionalNN() { }
		//kernelSizes j-hat sizes should be of size 3. This is not a 4d convolutional network.
		public ConvolutionalNN ConvolutionalNN(int[,] kernelSizes, int transitionElementAmount)
		{
			Random random = new Random();
			this.kernelSizes = kernelSizes;
			this.transitionElementAmount = transitionElementAmount;
			//when element amount falls below the transitionElementAmount, network will transition from
			//convolutional to fully connected.

			
		}	
	}
}