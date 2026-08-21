classdef PropInvoluteAlgebra < matlab.unittest.TestCase
    % PropInvoluteAlgebra: algebraic properties of involute over random signals

    methods(TestClassSetup)
        function setupTestClass(testCase)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), ...
                                 '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
        end
    end

    methods(Test)

        function testInvoluteDoubleApplication(testCase)
            % involute(involute(x)) = x exactly
            rng(42);
            num_trials = 100;

            for trial = 1:num_trials
                % Generate length once so real and imaginary parts have the same size.
                N = randi([32, 512]);
                x = randn(N, 1) + 1i*randn(N, 1);
                result = involute(involute(x));
                testCase.verifyEqual(result, x, 'AbsTol', 1e-12);
            end
        end

        function testInvoluteFFTRelation(testCase)
            % fft(involute(x)) = conj(fft(x)) within 1e-12
            rng(42);
            num_trials = 100;

            for trial = 1:num_trials
                N = randi([32, 512]);
                x = randn(N, 1) + 1i*randn(N, 1);
                lhs = fft(involute(x));
                rhs = conj(fft(x));
                testCase.verifyEqual(lhs, rhs, 'AbsTol', 1e-12);
            end
        end

        function testInvoluteConjugateCommutes(testCase)
            % involute(conj(x)) = conj(involute(x))
            rng(42);
            num_trials = 100;

            for trial = 1:num_trials
                N = randi([32, 512]);
                x = randn(N, 1) + 1i*randn(N, 1);
                lhs = involute(conj(x));
                rhs = conj(involute(x));
                testCase.verifyEqual(lhs, rhs, 'AbsTol', 1e-12);
            end
        end

        function testInvoluteOnRealFFT(testCase)
            % For real x: fft(involute(x)) = conj(fft(x))
            % This is the fundamental LTFAT property (docstring):
            %   conj(dft(f)) == dft(involute(f))
            % Note: involute(fft(x)) != conj(fft(x)) in general;
            %       the involute must be applied INSIDE the fft, not outside.
            rng(42);
            num_trials = 100;

            for trial = 1:num_trials
                N = randi([32, 512]);
                x = randn(N, 1);          % real signal
                lhs = fft(involute(x));   % fft of involuted signal
                rhs = conj(fft(x));       % conjugate of FFT
                testCase.verifyEqual(lhs, rhs, 'AbsTol', 1e-12);
            end
        end

    end

end
