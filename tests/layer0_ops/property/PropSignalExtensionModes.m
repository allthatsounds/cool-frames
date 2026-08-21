classdef PropSignalExtensionModes < matlab.unittest.TestCase
    % PropSignalExtensionModes: property tests for signal extension modes

    methods(TestClassSetup)
        function setupTestClass(testCase)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), ...
                                 '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
        end
    end

    methods(Test)

        function testPeriodicMode(testCase)
            % 'per': layout is [f(end-extLen+1:end) ; f ; f(1:extLen)]
            % Left extension = tail of f, right extension = head of f.
            rng(42);
            num_trials = 50;

            for trial = 1:num_trials
                L      = randi([32, 256]);
                f      = randn(L, 1);
                % Use floor(L/4) so extLen is always a valid integer
                extLen = randi([4, floor(L/4)]);

                f_ext = comp_extBoundary(f, extLen, 'per');

                % Total length correct
                testCase.verifyEqual(length(f_ext), L + 2*extLen, ...
                    sprintf('Trial %d: wrong total length', trial));
                % Left extension equals tail of f
                testCase.verifyEqual(f_ext(1:extLen), f(L-extLen+1:L), 'AbsTol', 1e-14, ...
                    sprintf('Trial %d: left extension wrong', trial));
                % Right extension equals head of f
                testCase.verifyEqual(f_ext(L+extLen+1:end), f(1:extLen), 'AbsTol', 1e-14, ...
                    sprintf('Trial %d: right extension wrong', trial));
            end
        end

        function testZpdMode(testCase)
            % 'zpd': extended region is exactly zero
            rng(42);
            num_trials = 50;

            for trial = 1:num_trials
                L      = randi([32, 256]);
                f      = randn(L, 1);
                extLen = randi([4, floor(L/4)]);

                f_ext = comp_extBoundary(f, extLen, 'zpd');

                left_ext  = f_ext(1:extLen);
                right_ext = f_ext(L+extLen+1:end);
                testCase.verifyEqual(left_ext,  zeros(extLen, 1), 'AbsTol', 1e-14, ...
                    sprintf('Trial %d: left extension not zero', trial));
                testCase.verifyEqual(right_ext, zeros(extLen, 1), 'AbsTol', 1e-14, ...
                    sprintf('Trial %d: right extension not zero', trial));
            end
        end

        function testSymMode(testCase)
            % 'sym': output has correct length
            rng(42);
            num_trials = 50;

            for trial = 1:num_trials
                L      = randi([32, 256]);
                f      = randn(L, 1);
                extLen = min(randi([4, floor(L/4)]), L-1);

                f_ext = comp_extBoundary(f, extLen, 'sym');

                testCase.verifyEqual(length(f_ext), L + 2*extLen, ...
                    sprintf('Trial %d: incorrect extended length', trial));
            end
        end

        function testExtensionWithZeroExtLen(testCase)
            % extLen = 0: output identical to input
            rng(42);
            num_trials = 30;

            for trial = 1:num_trials
                f     = randn(64, 1);
                f_ext = comp_extBoundary(f, 0, 'per');
                testCase.verifyEqual(length(f_ext), length(f));
                testCase.verifyEqual(f_ext, f, 'AbsTol', 1e-14);
            end
        end

        function testExtensionWithQuarterLength(testCase)
            % extLen = L/4 (integer): middle segment preserved, total length correct
            rng(42);
            num_trials = 30;

            for trial = 1:num_trials
                L      = 128;          % fixed so L/4 = 32 is exact
                f      = randn(L, 1);
                extLen = L / 4;

                modes = {'per', 'sym', 'zpd'};
                for idx = 1:length(modes)
                    f_ext = comp_extBoundary(f, extLen, modes{idx});
                    testCase.verifyEqual(length(f_ext), L + 2*extLen, ...
                        sprintf('Trial %d, mode %s: length mismatch', trial, modes{idx}));
                    testCase.verifyEqual(f_ext(extLen+1:extLen+L), f, 'AbsTol', 1e-14, ...
                        sprintf('Trial %d, mode %s: original signal not preserved', trial, modes{idx}));
                end
            end
        end

        function testAsymmetricSignal(testCase)
            % All three modes produce correct total length for random signals
            rng(42);
            num_trials = 50;

            for trial = 1:num_trials
                L      = randi([32, 256]);
                f      = randn(L, 1);
                extLen = randi([4, min(32, floor(L/4))]);

                modes = {'per', 'sym', 'zpd'};
                for idx = 1:length(modes)
                    f_ext = comp_extBoundary(f, extLen, modes{idx});
                    testCase.verifyEqual(length(f_ext), L + 2*extLen, ...
                        sprintf('Trial %d, mode %s: length mismatch', trial, modes{idx}));
                end
            end
        end

    end

end
