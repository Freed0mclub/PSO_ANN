using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PSO_ANN.ANN
{
    public class Layer
    {
        public Neuron[] Neurons { get; private set; }

        public Layer(int inputCount, int neuronCount)
        {
            Neurons = new Neuron[neuronCount];

            for (int i = 0; i < neuronCount; i++)
                Neurons[i] = new Neuron(inputCount);
        }

        public double[] Compute(double[] inputs)
        {
            double[] outputs = new double[Neurons.Length];

            for (int i = 0; i < Neurons.Length; i++)
                outputs[i] = Neurons[i].Activate(inputs);

            return outputs;
        }

        public int SetWeights(double[] weights, int startPos)
        {
            int pos = startPos;

            foreach (var neuron in Neurons)
            {
                for (int w = 0; w < neuron.Weights.Length; w++)
                    neuron.Weights[w] = weights[pos++];

                neuron.Bias = weights[pos++];
            }

            return pos;
        }

        public int GetWeightsCount()
        {
            int count = 0;

            foreach (var neuron in Neurons)
                count += neuron.Weights.Length + 1; // +1 for Bias

            return count;
        }
    }
}

