using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace neuralnet
{
    class ConvolutionalNN
    {
        //misc functions for convnet below


        //misc functions for fully connected neuralnet classifier below
        private double[] MatrixVectorProduct(double[,] matrix, double[] vector)
        {
            double[] final = new double[matrix.GetLength(0)];

            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                final[i] = 0;
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    final[i] += matrix[i, j] * vector[j];
                }
            }

            return final;
        }
        private double[] Add(double[] a, double[] b)
        {
            double[] final = new double[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                final[i] = a[i] + b[i];
            }
            return final;
        }

    }
}