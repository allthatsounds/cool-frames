classdef TestSignalExtension < matlab.unittest.TestCase
    % TestSignalExtension: unit tests for comp_extBoundary

    methods(TestClassSetup)
        function setupTestClass(testCase)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), ...
                                 '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
        end
    end

    methods(Test)

        function testExtBoundaryPeriodicMode(testCase)
            % 'per' mode: layout is [f(end-extLen+1:end) ; f ; f(1:extLen)]
            % Left extension  = tail of f,  right extension = head of f.
            rng(42);
            f = randn(64, 1);
            extLen = 16;
            f_ext = comp_extBoundary(f, extLen, 'per');

            L = length(f);
            % Left extension (positions 1:extLen) == last extLen samples of f
            testCase.verifyEqual(f_ext(1:extLen), f(L-extLen+1:L), 'AbsTol', 1e-14, ...
                'Periodic left extension does not match tail of f');
            % Right extension (positions L+extLen+1:end) == first extLen samples of f
            testCase.verifyEqual(f_ext(L+extLen+1:end), f(1:extLen), 'AbsTol', 1e-14, ...
                'Periodic right extension does not match head of f');
        end

        function testExtBoundaryZpdMode(testCase)
            % 'zpd' mode: extended region is zero
            rng(42);
            f = randn(64, 1);
            extLen = 16;
            f_ext = comp_extBoundary(f, extLen, 'zpd');

            L = length(f);
            testCase.verifyEqual(f_ext(1:extLen), zeros(extLen, 1), 'AbsTol', 1e-14);
            testCase.verifyEqual(f_ext(L+extLen+1:end), zeros(extLen, 1), 'AbsTol', 1e-14);
        end

        function testExtBoundarySymMode(testCase)
            % 'sym' mode: correct length
            rng(42);
            f = randn(64, 1);
            extLen = 16;
            f_ext = comp_extBoundary(f, extLen, 'sym');

            L = length(f);
            testCase.verifyEqual(length(f_ext), L + 2*extLen);
        end

        function testExtBoundaryLength(testCase)
            % Extension has correct total length: L + 2*extLen for all modes
            rng(42);
            f = randn(64, 1);
            modes = {'per', 'sym', 'symw', 'asym', 'asymw', 'sp0', 'zpd'};

            for mode = modes
                extLen = 16;
                f_ext = comp_extBoundary(f, extLen, mode{1});
                expected_len = length(f) + 2*extLen;
                testCase.verifyEqual(length(f_ext), expected_len, ...
                    sprintf('Mode %s: incorrect length', mode{1}));
            end
        end

        function testExtBoundaryZeroExtension(testCase)
            % extLen = 0: output identical to input
            rng(42);
            f = randn(64, 1);
            extLen = 0;
            f_ext = comp_extBoundary(f, extLen, 'per');

            testCase.verifyEqual(length(f_ext), length(f));
            testCase.verifyEqual(f_ext, f, 'AbsTol', 1e-14);
        end

        function testExtBoundaryOriginalPreserved(testCase)
            % Middle section (extLen+1 : extLen+L) always equals the original f
            rng(42);
            L = 128;
            f = randn(L, 1);
            extLen = L / 4;

            modes = {'per', 'sym', 'zpd'};
            for mode = modes
                f_ext = comp_extBoundary(f, extLen, mode{1});
                testCase.verifyEqual(length(f_ext), L + 2*extLen);
                testCase.verifyEqual(f_ext(extLen+1:extLen+L), f, 'AbsTol', 1e-14, ...
                    sprintf('Mode %s: original signal not preserved in the middle', mode{1}));
            end
        end

        function testExtBoundaryFullLength(testCase)
            % extLen = L: total output length = 3*L
            rng(42);
            L = 64;
            f = randn(L, 1);
            extLen = L;

            f_ext = comp_extBoundary(f, extLen, 'per');
            testCase.verifyEqual(length(f_ext), 3*L);
        end

        function testExtBoundaryAsymmetricSignal(testCase)
            % All modes work without error on a generic random signal
            rng(42);
            f = randn(64, 1);
            extLen = 8;

            modes = {'per', 'sym', 'symw', 'asym', 'asymw', 'sp0', 'zpd'};
            for mode = modes
                f_ext = comp_extBoundary(f, extLen, mode{1});
                testCase.verifyEqual(length(f_ext), 64 + 2*8, ...
                    sprintf('Mode %s: length mismatch', mode{1}));
            end
        end

        function testExtBoundaryAllModes(testCase)
            % All named modes run without exception
            rng(42);
            f = randn(64, 1);
            extLen = 8;

            modes = {'per', 'sym', 'symw', 'asym', 'asymw', 'sp0', 'zpd'};
            for mode = modes
                try
                    f_ext = comp_extBoundary(f, extLen, mode{1});
                    testCase.verifyEqual(length(f_ext), 64 + 2*8);
                catch ME
                    testCase.verifyFail(sprintf('Mode %s failed: %s', mode{1}, ME.message));
                end
            end
        end

    end

end
