//! A proportional-integral-derivative (PID) controller.
#![no_std]
extern crate num_traits;
use num_traits::{Bounded, Signed, Zero};

use core::ops::{Add, Mul, Sub};

pub trait State: Sub<Self, Output = Self> + Copy {}
pub trait Control: Bounded + Zero + Signed + Add + PartialOrd + Copy {}
pub trait Gain<S, C>: Copy + Mul<S, Output = C> {}

#[derive(Debug)]
pub struct Pid<S, C, G>
where
    S: State,
    C: Control,
    G: Gain<S, C>,
{
    /// Proportional gain.
    pub kp: G,
    /// Integral gain.
    pub ki: G,
    /// Derivative gain.
    pub kd: G,
    /// Limit of contribution of P term: `(-p_limit <= P <= p_limit)`
    pub p_limit: C,
    /// Limit of contribution of I term `(-i_limit <= I <= i_limit)`
    pub i_limit: C,
    /// Limit of contribution of D term `(-d_limit <= D <= d_limit)`
    pub d_limit: C,
    /// Limit of the sum of PID terms `(-limit <= P + I + D <+ limit)`
    pub limit: C,

    pub setpoint: S,
    prev_measurement: Option<S>,
    /// `integral_term = sum[error(t) * ki(t)] (for all t)`
    integral_term: C,
}

#[derive(Debug)]
pub struct ControlOutput<C> {
    /// Contribution of the P term to the output.
    pub p: C,
    /// Contribution of the I term to the output.
    /// `i = sum[error(t) * ki(t)] (for all t)`
    pub i: C,
    /// Contribution of the D term to the output.
    pub d: C,
    /// Output of the PID controller.
    pub output: C,
}

impl<S, C, G> Pid<S, C, G>
where
    S: State,
    C: Control,
    G: Gain<S, C>,
{
    pub fn new(kp: G, ki: G, kd: G, setpoint: S) -> Self {
        let max = C::max_value();
        Self {
            kp,
            ki,
            kd,
            p_limit: max,
            i_limit: max,
            d_limit: max,
            limit: max,
            setpoint,
            prev_measurement: None,
            integral_term: C::zero(),
        }
    }

    /// Resets the integral term back to zero. This may drastically change the
    /// control output.
    pub fn reset_integral_term(&mut self) {
        self.integral_term = C::zero();
    }

    fn bound(v: C, b: C) -> C {
        return (if v.abs() < b { v.abs() } else { b }) * v.signum();
    }

    /// Given a new measurement, calculates the next control output.
    pub fn next_control_output(&mut self, measurement: S) -> ControlOutput<C> {
        let error = self.setpoint - measurement;

        let p = Self::bound(self.kp * error, self.p_limit);

        // Mitigate output jumps when ki(t) != ki(t-1).
        // While it's standard to use an error_integral that's a running sum of
        // just the error (no ki), because we support ki changing dynamically,
        // we store the entire term so that we don't need to remember previous
        // ki values.
        // Mitigate integral windup: Don't want to keep building up error
        // beyond what i_limit will allow.
        self.integral_term = Self::bound(self.integral_term + self.ki * error, self.i_limit);

        // Mitigate derivative kick: Use the derivative of the measurement
        // rather than the derivative of the error.
        let d = match self.prev_measurement.as_ref() {
            Some(prev_measurement) => {
                Self::bound(self.kd * (*prev_measurement - measurement), self.d_limit)
            }
            None => C::zero(),
        };
        self.prev_measurement = Some(measurement);

        let output = Self::bound(p + self.integral_term + d, self.limit);

        ControlOutput {
            p,
            i: self.integral_term,
            d,
            output,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    impl Gain<f32, f32> for f32 {}
    impl State for f32 {}
    impl Control for f32 {}

    impl Gain<f64, f64> for f64 {}
    impl State for f64 {}
    impl Control for f64 {}

    #[test]
    fn proportional() {
        let mut pid: Pid<f32, f32, f32> = Pid::new(2.0, 0.0, 0.0, 10.0);
        assert_eq!(pid.setpoint, 10.0);

        // Test simple proportional
        assert_eq!(pid.next_control_output(0.0).output, 20.0);

        // Test proportional limit
        pid.p_limit = 10.0;
        assert_eq!(pid.next_control_output(0.0).output, 10.0);
    }

    #[test]
    fn derivative() {
        let mut pid: Pid<f32, f32, f32> = Pid::new(0.0, 0.0, 2.0, 10.0);

        // Test that there's no derivative since it's the first measurement
        assert_eq!(pid.next_control_output(0.0).output, 0.0);

        // Test that there's now a derivative
        assert_eq!(pid.next_control_output(5.0).output, -10.0);

        // Test derivative limit
        pid.d_limit = 5.0;
        assert_eq!(pid.next_control_output(10.0).output, -5.0);
    }

    #[test]
    fn integral() {
        let mut pid = Pid::new(0.0, 2.0, 0.0, 10.0);

        // Test basic integration
        assert_eq!(pid.next_control_output(0.0).output, 20.0);
        assert_eq!(pid.next_control_output(0.0).output, 40.0);
        assert_eq!(pid.next_control_output(5.0).output, 50.0);

        // Test limit
        pid.i_limit = 50.0;
        assert_eq!(pid.next_control_output(5.0).output, 50.0);
        // Test that limit doesn't impede reversal of error integral
        assert_eq!(pid.next_control_output(15.0).output, 40.0);

        // Test that error integral accumulates negative values
        let mut pid2 = Pid::new(0.0, 2.0, 0.0, -10.0);
        assert_eq!(pid2.next_control_output(0.0).output, -20.0);
        assert_eq!(pid2.next_control_output(0.0).output, -40.0);

        pid2.i_limit = 50.0;
        assert_eq!(pid2.next_control_output(-5.0).output, -50.0);
        // Test that limit doesn't impede reversal of error integral
        assert_eq!(pid2.next_control_output(-15.0).output, -40.0);
    }

    #[test]
    fn pid() {
        let mut pid = Pid::new(1.0, 0.1, 1.0, 10.0);

        let out = pid.next_control_output(0.0);
        assert_eq!(out.p, 10.0); // 1.0 * 10.0
        assert_eq!(out.i, 1.0); // 0.1 * 10.0
        assert_eq!(out.d, 0.0); // -(1.0 * 0.0)
        assert_eq!(out.output, 11.0);

        let out = pid.next_control_output(5.0);
        assert_eq!(out.p, 5.0); // 1.0 * 5.0
        assert_eq!(out.i, 1.5); // 0.1 * (10.0 + 5.0)
        assert_eq!(out.d, -5.0); // -(1.0 * 5.0)
        assert_eq!(out.output, 1.5);

        let out = pid.next_control_output(11.0);
        assert_eq!(out.p, -1.0); // 1.0 * -1.0
        assert_eq!(out.i, 1.4); // 0.1 * (10.0 + 5.0 - 1)
        assert_eq!(out.d, -6.0); // -(1.0 * 6.0)
        assert_eq!(out.output, -5.6);

        let out = pid.next_control_output(10.0);
        assert_eq!(out.p, 0.0); // 1.0 * 0.0
        assert_eq!(out.i, 1.4); // 0.1 * (10.0 + 5.0 - 1.0 + 0.0)
        assert_eq!(out.d, 1.0); // -(1.0 * -1.0)
        assert_eq!(out.output, 2.4);
    }

    #[test]
    fn f32_and_f64() {
        let mut pid32 = Pid::new(2.0f32, 0.0, 0.0, 10.0);

        let mut pid64: Pid<f64, f64, f64> = Pid::new(2.0f64, 0.0, 0.0, 10.0);

        assert_eq!(
            pid32.next_control_output(0.0).output,
            pid64.next_control_output(0.0).output as f32
        );
        assert_eq!(
            pid32.next_control_output(0.0).output as f64,
            pid64.next_control_output(0.0).output
        );
    }
}
