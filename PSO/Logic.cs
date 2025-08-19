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
                // Initialize positions in [-initPosRange, +initPosRange]
                Position[d] = (_rng.NextDouble() * 2 - 1) * initPosRange;
                // Initialize velocities in [-initVelRange, +initVelRange]
                Velocity[d] = (_rng.NextDouble() * 2 - 1) * initVelRange;
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

    public class Swarm
    {
        public Particle[] Particles { get; }
        public double[] GlobalBestPosition { get; protected set; }
        public double GlobalBestFitness { get; protected set; }

        // PSO hyperparameters
        public double Inertia { get; set; } = 0.729;
        public double CognitiveFactor { get; set; } = 1.49445;
        public double SocialFactor { get; set; } = 1.49445;

        // Velocity clamping
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


        // Performs one iteration of PSO: evaluations, best updates, velocity & position updates.
  
        // <param name="fitnessFunc">Function that returns fitness for a given position vector.</param>
        public void UpdateParticles(Func<double[], double> fitnessFunc)
        {
            // 1) Evaluate fitness for each particle and update personal & global bests
            foreach (var p in Particles)
            {
                double fitness = fitnessFunc(p.Position);
                p.UpdatePersonalBest(fitness);

                if (fitness < GlobalBestFitness)
                {
                    GlobalBestFitness = fitness;
                    Array.Copy(p.Position, GlobalBestPosition, p.Position.Length);
                }
            }

            // 2) Update velocity and position for each particle
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

                    // Clamp velocity
                    vel = Math.Max(-MaxVelocity, Math.Min(MaxVelocity, vel));
                    p.Velocity[d] = vel;

                    // Update position
                    p.Position[d] += vel;
                }
            }
        }
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
}


