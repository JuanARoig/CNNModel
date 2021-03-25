using System;

namespace neuralnet
{
    public partial class ConvolutionalNN
    {
        protected double[,,] REluOperation(double[,,] inputTensor)
        {
            if (inputTensor == null)
            {
                throw new Exception("Null tensor");
            }
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
                            final[i, j, k] = 0;
                        }
                    }
                }
            }
            return final;
        }
  
        protected double[] SoftmaxOperation(double[] inputVector)//activation function for fully connected portion
        {
            if(inputVector.Length == 0)
            {
                throw new Exception("Null vector");
            }
            double sum = 0;
            double[] final = new double[inputVector.Length];
            for (int i = 0; i < inputVector.Length; i++)
            {
                sum += inputVector[i];
            }
            for (int i = 0; i < inputVector.Length; i++)
            {
                final[i] = Math.Exp(inputVector[i]) / sum;
            }
            return final;
        }
    }
}