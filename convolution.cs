using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace neuralnet
{
    class ConvolutionalNN
    {
        private double[,,] ConvolutionProduct(double[,,] tensor1, double[,,] kernel) 
	    {	
	    	sizeI = 0;
	    	sizeJ = 0;
	    	sizeK = 0;

            //the dimensions of the resulting matrix are funky.
	    	if (tensor1.GetLength(0) - kernel.GetLength(0) == 1) 
	    	{
	    		sizeI += 2;
	    	} 
	    	else 
	    	{
	    		sizeI += (tensor1.GetLength(0)-kernel.GetLength(0)) + 1;
	    	}
	    	if (tensor1.GetLength(1) - kernel.GetLength(1) == 1)
	    	{
	    		sizeJ += 2;
	    	}
	    	else
	    	{
	    		sizeJ += (tensor1.GetLength(1) - kernel.GetLength(1)) + 1;
	    	}
	    	if (tensor1.GetLength(2) - kernel.GetLength(2) == 1)
	    	{
	    		sizeK += 2;
	    	}
	    	else
	    	{
	    		sizeK += (tensor1.GetLength(2) - kernel.GetLength(2)) + 1;
	    	}

	    	double[,,] tensorFinal = new double[sizeI,sizeJ,sizeK];
		    for (int i = 0; i < tensorFinal.GetLength(0); i++)
	    	{
		    	for(int j = 0; j < tensorFinalGetLength(1); j++)	
		    	{	
		    		for(int k = 0; k < tensorFinal.GetLength(2); k++)
		    		{
		    			tensorFinal[i, j, k] = 0;
		    		}
		    	}
	    	}
		
	    	for(int i = 0; i < tensor1.GetLength(0) - kernel.GetLength(0) - 1; i++)
	    	{
			    for(int j = 0; j < tensor1.GetLength(1) - kernel.GetLength(1) - 1; j++)
		    	{
				    for(int k = 0; k < tensor1.GetLength(2) - kernel.GetLength(2) - 1; k++)
				    {
					    int startA = i;
					    int startB = j;
					    int startC = k;
					    int r = kernel1Size.GetLength(0);
					    int s = kernel1Size.GetLength(1);
					    int t = kernel1Size.GetLength(2);
					    for(int a = i; a < startA + r; a++)
					    {
					    	for (int b = j; b < startB + s; b++)
						    {
							    for (int c = k; c < startC + t; c++)
						    	{
							    	tensorFinal[i, j, k] += tensor1[a, b, c]*kernel[a-startA, b-startB, c-startC];
							    }
						    }
					    }				
				    }
			    }
		    }
		    return tensorFinal;	
	    }	
    }
}