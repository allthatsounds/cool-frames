classdef TestMathUtils < matlab.unittest.TestCase
%TESTMATHUTILS  Unit tests for mathematical utility functions.
%
%   Covers: pderiv, psech
%
%   pderiv(f)            -- periodic numerical/spectral derivative
%   psech(L)             -- periodized hyperbolic-secant window

    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), ...
                                 '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
        end
    end

    % ── pderiv: output dimensions ─────────────────────────────────────────
    methods (Test)

        function testPderivOutputLengthVector(tc)
            % pderiv must return a vector of the same length as input.
            x = randn(64, 1);
            fd = pderiv(x);
            tc.verifyEqual(numel(fd), numel(x), ...
                'pderiv: output length must equal input length.');
        end

        function testPderivOutputSizeMatrix(tc)
            % When applied to a matrix, pderiv acts along columns by default.
            X = randn(64, 4);
            FD = pderiv(X);
            tc.verifyEqual(size(FD), size(X), ...
                'pderiv: output size must equal input size for a matrix.');
        end

        function testPderivDerivativeOfConstantIsZero(tc)
            % Derivative of a constant signal is zero (all orders).
            L = 64;
            f = ones(L, 1);
            for order = [2, 4, Inf]
                fd = pderiv(f, [], order);
                tc.verifyLessThan(norm(fd), 1e-10, ...
                    sprintf('pderiv: derivative of constant must be zero (order %g).', order));
            end
        end

        function testPderivLinearSine(tc)
            % Derivative of sin(2*pi*k*x) = 2*pi*k * cos(2*pi*k*x).
            % On [0,1) with L samples: f(n) = sin(2*pi*n/L),
            % f'(n) = (2*pi/L) * cos(2*pi*n/L) scaled by L because pderiv
            % returns the derivative on [0,1) (not [0,L)).
            % pderiv returns d/d(n/L), so derivative = 2*pi * cos(2*pi*n/L).
            L = 128;
            n = (0:L-1).';
            f  = sin(2*pi*n/L);
            fd = pderiv(f, [], Inf);          % spectral derivative
            expected = 2*pi * cos(2*pi*n/L);
            rel_err = norm(fd - expected) / norm(expected);
            tc.verifyLessThan(rel_err, 1e-10, ...
                'pderiv(Inf): derivative of sin must equal 2*pi*cos (spectral).');
        end

        function testPderivOrderInf(tc)
            % Spectral derivative (order=Inf) of a real signal is real.
            f = randn(64, 1);
            fd = pderiv(f, [], Inf);
            tc.verifyTrue(isreal(fd), ...
                'pderiv(Inf): derivative of a real signal must be real.');
        end

        function testPderivOrder2VsOrder4Agreement(tc)
            % For a smooth low-frequency signal, orders 2 and 4 give close results.
            L = 256;
            n = (0:L-1).';
            f = sin(2*pi*3*n/L) + 0.5*cos(2*pi*5*n/L);
            fd2 = pderiv(f, [], 2);
            fd4 = pderiv(f, [], 4);
            rel_err = norm(fd2 - fd4) / (norm(fd4) + eps);
            tc.verifyLessThan(rel_err, 0.02, ...
                'pderiv: orders 2 and 4 must agree within 2% for smooth signals.');
        end

        function testPderivRealOutputForRealInput(tc)
            % All finite difference orders must return real output for real input.
            f = randn(64, 1);
            for order = [2, 4]
                fd = pderiv(f, [], order);
                tc.verifyTrue(isreal(fd), ...
                    sprintf('pderiv(order=%d): real input must give real output.', order));
            end
        end

    end

    % ── psech: output dimensions and basic properties ─────────────────────
    methods (Test)

        function testPsechOutputLength(tc)
            % psech(L) must return a column vector of length L.
            for L = [32, 64, 128, 256]
                g = psech(L);
                tc.verifyEqual(numel(g), L, ...
                    sprintf('psech: output length must equal L=%d.', L));
                tc.verifyEqual(size(g, 2), 1, ...
                    sprintf('psech: output must be a column vector (L=%d).', L));
            end
        end

        function testPsechUnitNorm(tc)
            % psech is normalized so that norm(g) = 1.
            for L = [32, 64, 128]
                g = psech(L);
                tc.verifyEqual(norm(g), 1, 'AbsTol', 1e-4, ...
                    sprintf('psech: norm must equal 1 for L=%d.', L));
            end
        end

        function testPsechIsRealValued(tc)
            % psech must return a real-valued signal.
            g = psech(64);
            tc.verifyTrue(isreal(g), 'psech: output must be real-valued.');
        end

        function testPsechDFTInvariance(tc)
            % The canonical psech with tfr=1 is its own DFT:
            % norm(g - dft(g)) should be close to zero.
            L = 128;
            g = psech(L);
            err = norm(g - dft(g));
            tc.verifyLessThan(err, 1e-10, ...
                'psech: canonical psech(L) must be its own DFT (DFT-invariant).');
        end

        function testPsechWholePointEven(tc)
            % psech is whole-point even: fft(psech(L)) is real.
            for L = [32, 64, 128]
                g = psech(L);
                G = fft(g);
                tc.verifyLessThan(max(abs(imag(G))), 1e-12, ...
                    sprintf('psech: fft must be real-valued (whole-point even), L=%d.', L));
            end
        end

        function testPsechTfrScaling(tc)
            % With tfr > 1, the window should be wider (higher concentration in time).
            L = 128;
            g1 = psech(L, 1);
            g2 = psech(L, 4);
            % Wider window in time → lower peak value
            tc.verifyLessThan(max(abs(g2)), max(abs(g1)), ...
                'psech: larger tfr must give a wider (higher-peak) window.');
        end

    end

end
