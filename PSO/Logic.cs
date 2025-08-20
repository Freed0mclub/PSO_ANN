using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using System;

namespace PSO_ANN.PSO
{

    /// Represents a single particle in the PSO swarm.

    public class Particle
    {
        public double[] Position { get; private set; }
        public double[] Velocity { get; private set; }
        public double[] BestPosition { get; private set; }
        public double BestFitness { get; set; }

        private static readonly Random _rng = new Random();

        public Particle(int dimensions, double initPosRange = 1.0, double initVelRange = 0.1)
        {
            Position = new double[dimensions];
            Velocity = new double[dimensions];
            BestPosition = new double[dimensions];
            BestFitness = double.MaxValue;

            for (int d = 0; d < dimensions; d++)
            {
                Position[d] = (_rng.NextDouble() * 2 - 1) * initPosRange;   // [-range, +range]
                Velocity[d] = (_rng.NextDouble() * 2 - 1) * initVelRange;   // [-range, +range]
                BestPosition[d] = Position[d];
            }
        }


        /// Update personal best if current fitness is better.

        public void UpdatePersonalBest(double fitness)
        {
            if (fitness < BestFitness)
            {
                BestFitness = fitness;
                Array.Copy(Position, BestPosition, Position.Length);
            }
        }
    }

    /// Manages a swarm of standard PSO  particles optimizing a given fitness function.

    /// Standard PSO swarm (no gradient logic here).
   
    public class Swarm
    {
        public Particle[] Particles { get; }

        // NOTE: protected setters allow derived classes (HybridSwarm) to update these.
        public double[] GlobalBestPosition { get; protected set; }
        public double GlobalBestFitness { get; protected set; }

        // PSO hyperparameters
        public double Inertia { get; set; } = 0.729;    // ω
        public double CognitiveFactor { get; set; } = 1.49445;  // c1
        public double SocialFactor { get; set; } = 1.49445;  // c2

        // Velocity clamping (optional safety)
        public double MaxVelocity { get; set; } = 0.5;

        private static readonly Random _rng = new Random();

        public Swarm(int particleCount, int dimensions)
        {
            Particles = new Particle[particleCount];
            GlobalBestPosition = new double[dimensions];
            GlobalBestFitness = double.MaxValue;

            for (int i = 0; i < particleCount; i++)
                Particles[i] = new Particle(dimensions);
        }


        /// One PSO iteration: evaluate fitness, update personal/global bests, then velocity/positions.

        // <param name="fitnessFunc">Function that returns fitness for a given position vector.</param>
        public void UpdateParticles(Func<double[], double> fitnessFunc)
        {
            // 1) Evaluate fitness and refresh bests
            foreach (var p in Particles)
            {
                double fit = fitnessFunc(p.Position);
                p.UpdatePersonalBest(fit);
                if (fit < GlobalBestFitness)
                {
                    GlobalBestFitness = fit;
                    Array.Copy(p.Position, GlobalBestPosition, p.Position.Length);
                }
            }

            // 2) Velocity and position updates
            foreach (var p in Particles)
            {
                for (int d = 0; d < p.Position.Length; d++)
                {
                    double r1 = _rng.NextDouble();
                    double r2 = _rng.NextDouble();

                    // PSO velocity update
                    double vel = Inertia * p.Velocity[d]
                                 + CognitiveFactor * r1 * (p.BestPosition[d] - p.Position[d])
                                 + SocialFactor * r2 * (GlobalBestPosition[d] - p.Position[d]);
                    // clamp
                    if (vel > MaxVelocity) vel = MaxVelocity;
                    if (vel < -MaxVelocity) vel = -MaxVelocity;

                    // Clamp velocity
                    vel = Math.Max(-MaxVelocity, Math.Min(MaxVelocity, vel));
                    p.Velocity[d] = vel;

                    // Update position
                    p.Position[d] += vel;
                }
            }
        }
    }

    
    /// Hybrid (memetic) PSO: normal PSO steps plus periodic local GD on elite particles.
    /// No gradient is stored inside particles.
   
        public class HybridSwarm : Swarm
        {
        public double GDLearningRate { get; set; } = 0.01; // η
        public int GDSteps { get; set; } = 3;    // inner GD steps per refinement
        public int GDEliteCount { get; set; } = 5;    // number of particles refined
        public int GDPeriod { get; set; } = 10;   // perform GD every N PSO iterations

            private int _iter = 0;
            private static readonly Random _rng = new Random();

            public HybridSwarm(int particleCount, int dimensions)
                : base(particleCount, dimensions) { }


        /// One hybrid iteration: PSO update + (periodically) a few GD steps on the top-k particles.

        /// <param name="fitnessFunc">Fitness function (lower is better).</param>
        /// <param name="gradientFunc">Returns gradient of fitness at given weights.</param>
            public void UpdateParticlesHybrid(
                Func<double[], double> fitnessFunc,
            Func<double[], double[]> gradientFunc)
            {
            // 1) Global exploration
                base.UpdateParticles(fitnessFunc);
                _iter++;

            // 2) Periodic local exploitation via GD
            if (gradientFunc == null) return;
            if (GDPeriod > 0 && (_iter % GDPeriod != 0)) return;

            var elites = Particles
                    .Select(p => new { P = p, Fit = fitnessFunc(p.Position) })
                    .OrderBy(a => a.Fit)
                    .Take(Math.Min(GDEliteCount, Particles.Length))
                    .ToArray();

            foreach (var e in elites)
                {
                    var p = e.P;
                    // work on a local copy of weights
                    var w = (double[])p.Position.Clone();

                    for (int s = 0; s < GDSteps; s++)
                    {
                        var g = gradientFunc(w);
                        for (int d = 0; d < w.Length; d++)
                            w[d] -= GDLearningRate * g[d];
                    }

                    double newFit = fitnessFunc(w);
                    if (newFit < e.Fit)
                    {
                    // accept improvement; reset velocity to avoid overshoot
                        Array.Copy(w, p.Position, w.Length);
                    Array.Clear(p.Velocity, 0, p.Velocity.Length);
                        p.UpdatePersonalBest(newFit);

                        if (newFit < GlobalBestFitness)
                        {
                            GlobalBestFitness = newFit;
                            Array.Copy(w, GlobalBestPosition, w.Length);
                        }
                    }
                }
            }
        }
}


        /*
         * commenting this approach out for now, as it is not used in the current implementation.
         * 
        public class HybridParticle : Particle
        {
            /// Stores gradient information for hybrid optimization.
            public double[] Gradient { get; private set; }

            public HybridParticle(int dimensions, double initPosRange = 1.0, double initVelRange = 0.1)
                : base(dimensions, initPosRange, initVelRange)
            {
                Gradient = new double[dimensions];
            }
        }
        
    }

    /// Manages a swarm of hybrid particles that combine PSO with gradient descent.

    public class HybridSwarm : Swarm
    {
        public double LearningRate { get; set; } = 0.01;
        public double Epsilon { get; set; } = 1e-4;
        private static readonly Random _rng = new Random();

        public HybridSwarm(int particleCount, int dimensions)
            : base(particleCount, dimensions)
        {
            // Replace standard particles with hybrid ones
            for (int i = 0; i < Particles.Length; i++)
                Particles[i] = new HybridParticle(dimensions);
        }


        /// Performs one hybrid iteration: compute gradients, then PSO+GD updates.

        public void UpdateParticlesHybrid(Func<double[], double> fitnessFunc)
        {
            // 1) Compute gradient for each hybrid particle
            foreach (HybridParticle p in Particles)
            {
                for (int d = 0; d < p.Position.Length; d++)
                {
                    double orig = p.Position[d];
                    p.Position[d] = orig + Epsilon;
                    double up = fitnessFunc(p.Position);
                    p.Position[d] = orig - Epsilon;
                    double down = fitnessFunc(p.Position);
                    p.Gradient[d] = (up - down) / (2 * Epsilon);
                    p.Position[d] = orig;
                }
            }

            // 2) PSO + gradient descent update
            foreach (HybridParticle p in Particles)
            {
                for (int d = 0; d < p.Position.Length; d++)
                {
                    double r1 = _rng.NextDouble();
                    double r2 = _rng.NextDouble();

                    // PSO component
                    double psoVel = Inertia * p.Velocity[d]
                                  + CognitiveFactor * r1 * (p.BestPosition[d] - p.Position[d])
                                  + SocialFactor * r2 * (GlobalBestPosition[d] - p.Position[d]);

                    // Gradient descent component
                    double gradVel = -LearningRate * p.Gradient[d];

                    // Combine and move
                    double vel = psoVel + gradVel;
                    p.Velocity[d] = vel;
                    p.Position[d] += vel;
                }
            }

            // 3) Evaluate and update bests
            foreach (HybridParticle p in Particles)
            {
                double fit = fitnessFunc(p.Position);
                p.UpdatePersonalBest(fit);
                if (fit < GlobalBestFitness)
                {
                    GlobalBestFitness = fit;
                    Array.Copy(p.Position, GlobalBestPosition, p.Position.Length);
                }
            }
        }
        
    }
        */




