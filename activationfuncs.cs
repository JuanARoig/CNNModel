using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace neuralnet
{
    class ConvolutionalNN
    {
        private double[,,] REluOperation(double[,,] inputTensor)
        {
            double[,,] final = new double[inputTensor.GetLength(0), inputTensor.GetLength(1), inputTensor.GetLength(2)];
            for (int i = 0; i < inputTensor.GetLength(0); i++)
            {
                for(int j = 0; j < inputTensor.GetLength(1); j++)
                {
                    for (int k = 0; k < inputTensor.GetLength(2); k++)
                    {
                        if(inputTensor[i, j, k] > 0)
                        {
                            final[i, j, k] = inputTensor[i, j, k];
                        }
                        else
                        {
                            final[i, j, k] = inputTensor[i, j, k];
                        }
                    }
                }
            }
            return final;
        }

          public double[] SoftmaxOperation(double[] inputVector)//activation function for fully connected portion
        {
            
        }
    }
}