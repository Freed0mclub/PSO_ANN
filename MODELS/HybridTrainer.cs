
using System;
using System.Linq;
using PSO_ANN.ANN;
using PSO_ANN.PSO;
using PSO_ANN.UTILS;


namespace PSO_ANN.MODELS
{
    /// <summary>
    /// Trainer that couples standard PSO with periodic local GD refinement (no gradient stored in particles).
    /// </summary>
    public static class HybridTrainer
    {
        public static (double[] bestWeights, double trainMse, double valMse) Run(
        string csvPath,
        int hiddenNeurons = 10,
        int particleCount = 50,
        int iterations = 1000,
        double psoInertia = 0.729,
        double psoCog = 1.49445,
        double psoSoc = 1.49445,
        // GD refinement controls
        int gdPeriod = 10,
        int gdSteps = 3,
        int gdElite = 5,
        double gdLR = 0.01,
        int gradBatch = 64,
        double epsilon = 1e-4,
        Action<int, double>? onIter = null, int logEvery = 1) // log every N iterations
        {
            // 1) Load & normalize
            var data = DataLoader.LoadAndNormalize(csvPath);
            int nFeat = data[0].inputs.Length;


            // 2) Split
            var rng = new Random(42);
            var shuffled = data.OrderBy(_ => rng.Next()).ToList();
            int split = (int)(0.8 * shuffled.Count);
            var train = shuffled.Take(split).ToList();
            var val = shuffled.Skip(split).ToList();


            // 3) Build ANN & swarm
            var ann = new NeuralNetwork(new[] { nFeat, hiddenNeurons, 1 });
            int dims = ann.GetWeightsCount();


            var swarm = new HybridSwarm(particleCount, dims)
            {
                Inertia = psoInertia,
                CognitiveFactor = psoCog,
                SocialFactor = psoSoc,
                GDLearningRate = gdLR,
                GDSteps = gdSteps,
                GDEliteCount = gdElite,
                GDPeriod = gdPeriod
            };


            // 4) Fitness on full train set (robust)
            double Fitness(double[] w)
            {
                ann.SetWeights(w);
                double s = 0;
                foreach (var (x, t) in train)
                {
                    var y = ann.Forward(x);
                    s += Math.Pow(y[0] - t[0], 2);
                }
                return s / train.Count;
            }


            // 5) Mini-batch numerical gradient (cheap and generic). Swap with backprop later.
            var batch = train.OrderBy(_ => rng.Next()).Take(Math.Min(gradBatch, train.Count)).ToArray();


            double BatchLoss(double[] w)
            {
                ann.SetWeights(w);
                double s = 0;
                foreach (var (x, t) in batch)
                {
                    var y = ann.Forward(x);
                    s += Math.Pow(y[0] - t[0], 2);
                }
                return s / batch.Length;
            }


            double[] NumericalGrad(double[] w)
            {
                // resample a mini-batch every time we compute a gradient
                var mini = train.OrderBy(_ => rng.Next())
                                .Take(Math.Min(gradBatch, train.Count))
                                .ToArray();

                double BatchLossLocal(double[] ww)
                {
                    ann.SetWeights(ww);
                    double s = 0;
                    foreach (var (x, t) in mini)
                    {
                        var y = ann.Forward(x);
                        s += Math.Pow(y[0] - t[0], 2);
                    }
                    return s / mini.Length;
                }

                var g = new double[w.Length];
                double f0 = BatchLossLocal(w);
                for (int d = 0; d < w.Length; d++)
                {
                    double tmp = w[d];
                    double step = epsilon * (1.0 + Math.Abs(tmp));
                    w[d] = tmp + step;
                    double f1 = BatchLossLocal(w);
                    g[d] = (f1 - f0) / step;
                    w[d] = tmp;
                }
                return g;
            }


            // 6) Hybrid loop
            for (int it = 1; it <= iterations; it++)
            {
                swarm.UpdateParticlesHybrid(Fitness, NumericalGrad);
                if (onIter != null && (it % logEvery == 0 || it == 1))
                    onIter(it, swarm.GlobalBestFitness);
            }

            // 7) Final metrics
            double trainMse = swarm.GlobalBestFitness;
            ann.SetWeights(swarm.GlobalBestPosition);
            double valMse = val.Sum(d => Math.Pow(ann.Forward(d.inputs)[0] - d.targets[0], 2)) / val.Count;


            return (swarm.GlobalBestPosition, trainMse, valMse);
        }
    }
}

/*
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
*/

