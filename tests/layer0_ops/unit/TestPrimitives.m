classdef TestPrimitives < matlab.unittest.TestCase
    % TestPrimitives: unit tests for involute, modcent, fftindex, floor23

    methods(TestClassSetup)
        function setupTestClass(testCase)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), ...
                                 '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
        end
    end

    methods(Test)

        function testInvoluteDoublingIsIdentity(testCase)
            % involute(involute(x)) = x for random complex x
            for trial = 1:10
                x = randn(100, 1) + 1i*randn(100, 1);
                result = involute(involute(x));
                testCase.verifyEqual(result, x, 'AbsTol', 1e-12);
            end
        end

        function testInvoluteFFTRelation(testCase)
            % fft(involute(x)) = conj(fft(x)) within 1e-12
            for trial = 1:10
                x = randn(128, 1) + 1i*randn(128, 1);
                lhs = fft(involute(x));
                rhs = conj(fft(x));
                testCase.verifyEqual(lhs, rhs, 'AbsTol', 1e-12);
            end
        end

        function testInvoluteDCElement(testCase)
            % involute(x)(1) = conj(x(1))
            % By definition: involute(x)(n) = conj(x(mod(-n,L)+1))
            % For n=1 (0-indexed n=0): mod(0,L)+1 = 1, then conjugated.
            % So the DC element is conjugated, not preserved.
            for trial = 1:10
                x = randn(100, 1) + 1i*randn(100, 1);
                inv_x = involute(x);
                testCase.verifyEqual(inv_x(1), conj(x(1)), 'AbsTol', 1e-14);
            end
        end

        function testModcentRange(testCase)
            % modcent output always in [-r/2, r/2) for random x and r
            rng(42);
            for trial = 1:20
                x = randn(1, 10);
                r_vals = [2, 4, 8, 16, 32];
                for r = r_vals
                    result = modcent(x, r);
                    testCase.verifyTrue(all(result >= -r/2 & result < r/2), ...
                        sprintf('Trial %d: modcent not in [-r/2, r/2)', trial));
                end
            end
        end

        function testModcentPeriodicity(testCase)
            % modcent(x + r, r) = modcent(x, r) (periodicity)
            rng(42);
            for trial = 1:20
                x = randn(1, 10);
                r = 16;
                result1 = modcent(x, r);
                result2 = modcent(x + r, r);
                testCase.verifyEqual(result1, result2, 'AbsTol', 1e-14);
            end
        end

        function testFFTIndexRange(testCase)
            % fftindex: output in [-ceil(N/2)+1, floor(N/2)] for various N
            N_vals = [8, 16, 32, 1024];
            for N = N_vals
                idx = fftindex(N);
                testCase.verifyTrue(all(idx >= -ceil(N/2)+1 & idx <= floor(N/2)), ...
                    sprintf('N=%d: fftindex out of range', N));
            end
        end

        function testFFTIndexLastElement(testCase)
            % For even N >= 4, fftindex returns [0:N/2, -N/2+1:-1],
            % so the LAST element is -1 (the negative-frequency bin just below DC).
            % The Nyquist bin (N/2) sits in the interior, not at the end.
            N_vals = [8, 16, 32, 1024];
            for N = N_vals
                idx = fftindex(N);
                testCase.verifyEqual(idx(end), -1, ...
                    sprintf('N=%d: last element should be -1', N));
            end
        end

        function testFloor23Result(testCase)
            % floor23: result <= input
            rng(42);
            for trial = 1:30
                input = randi([1, 10000]);
                result = floor23(input);
                testCase.verifyLessThanOrEqual(result, input);
            end
        end

        function testFloor23SmoothNumber(testCase)
            % floor23: result factors as 2^i * 3^j
            rng(42);
            for trial = 1:30
                input = randi([1, 10000]);
                result = floor23(input);

                val = result;
                while mod(val, 2) == 0
                    val = val / 2;
                end
                while mod(val, 3) == 0
                    val = val / 3;
                end
                testCase.verifyEqual(val, 1, ...
                    sprintf('floor23(%d)=%d is not 2-3 smooth', input, result));
            end
        end

        function testFloor23LowerBound(testCase)
            % floor23: result > input/6
            rng(42);
            for trial = 1:30
                input = randi([100, 10000]);
                result = floor23(input);
                testCase.verifyGreaterThan(result, input/6, ...
                    sprintf('floor23(%d)=%d is too small', input, result));
            end
        end

    end

end
