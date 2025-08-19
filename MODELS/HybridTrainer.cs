using System;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using PSO_ANN.ANN;
using PSO_ANN.PSO;
using PSO_ANN.UTILS;

namespace PSO_ANN.MODELS
{

    /// Runs hybrid PSO+gradient-descent training on an ANN using HybridSwarm.

    public static class HybridTrainer
    {

        /// Executes hybrid optimization using HybridSwarm and HybridParticle.
        /// </summary>
        /// <param name="csvPath">Path to space-delimited CSV file.</param>
        /// <param name="hiddenNeurons">Hidden layer size.</param>
        /// <param name="particleCount">Number of swarm particles.</param>
        /// <param name="iterations">Iterations for hybrid update.</param>
        /// <param name="learningRate">Step size for gradient component.</param>
        /// <param name="epsilon">Finite-difference epsilon for gradient estimation.</param>
        /// <returns>Best weights, training MSE, validation MSE.</returns>
        public static (double[] bestWeights, double trainMse, double valMse) Run(
            string csvPath,
            int hiddenNeurons = 10,
            int particleCount = 50,
            int iterations = 1000,
            double learningRate = 0.01,
            double epsilon = 1e-4)
        {
            // 1) Load & normalize data
            var data = DataLoader.LoadAndNormalize(csvPath);
            int nFeatures = data[0].inputs.Length;

            // 2) Train/validation split
            var rnd = new Random(42);
            var shuffled = data.OrderBy(_ => rnd.Next()).ToList();
            int splitIndex = (int)(0.8 * shuffled.Count);
            var trainSet = shuffled.Take(splitIndex).ToList();
            var valSet = shuffled.Skip(splitIndex).ToList();

            // 3) Build the ANN
            var topology = new[] { nFeatures, hiddenNeurons, 1 };
            var ann = new NeuralNetwork(topology);
            int dims = ann.GetWeightsCount();

            // 4) Initialize HybridSwarm
            var swarm = new HybridSwarm(particleCount, dims)
            {
                Inertia = 0.729,
                CognitiveFactor = 1.49445,
                SocialFactor = 1.49445,
                LearningRate = learningRate,
                Epsilon = epsilon
            };

            // 5) Define fitness (mean squared error on training set)
            Func<double[], double> fitness = weights =>
            {
                ann.SetWeights(weights);
                double sumSq = 0;
                foreach (var (inputs, targets) in trainSet)
                {
                    var output = ann.Forward(inputs);
                    sumSq += Math.Pow(output[0] - targets[0], 2);
                }
                return sumSq / trainSet.Count;
            };

            // 6) Run hybrid PSO+GD optimization
            for (int i = 0; i < iterations; i++)
            {
                swarm.UpdateParticlesHybrid(fitness);
            }

            // 7) Capture training MSE
            double trainMse = swarm.GlobalBestFitness;

            // 8) Evaluate on validation set
            ann.SetWeights(swarm.GlobalBestPosition);
            double valMse = valSet.Sum(d =>
            {
                var o = ann.Forward(d.inputs);
                return Math.Pow(o[0] - d.targets[0], 2);
            }) / valSet.Count;

            // 9) Return best weights and errors
            return (swarm.GlobalBestPosition, trainMse, valMse);
        }
    }
}
