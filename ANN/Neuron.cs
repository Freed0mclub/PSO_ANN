using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PSO_ANN.ANN
{
    public class Neuron
    {
        public double[] Weights { get; set; }
        public double Bias { get; set; }

        public Neuron(int inputCount)
        {
            Weights = new double[inputCount];
            Bias = 0.0;
            InitializeWeights();
        }

        private void InitializeWeights()
        {
            Random rnd = new Random();
            for (int i = 0; i < Weights.Length; i++)
                Weights[i] = rnd.NextDouble() * 2 - 1; // weights initialized between [-1, 1]
        }

        public double Activate(double[] inputs)
        {
            double sum = Bias;
            for (int i = 0; i < Weights.Length; i++)
                sum += inputs[i] * Weights[i];

            return Sigmoid(sum);
        }

        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }
    }

}
