classdef PropParsevalConsistency < matlab.unittest.TestCase
    % PropParsevalConsistency: property tests for energy distribution in subbands
    %
    % For a tight frame with bound A:  sum_m (1/a_m) * ||c_m||^2 = A * ||x||^2
    % For audfilters (approximate tight frame), the ratio should be close to 1
    % when weighted by 1/a_m.
    %
    % Uses the public filterbank() API.

    properties
        Ls
        fs
        noise_real
        noise_complex
        g
        a
    end

    methods(TestClassSetup)
        function setupTestClass(testCase)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), ...
                                 '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
            rng(42);
            testCase.Ls = 1024;
            testCase.fs = 8000;
            [testCase.g, testCase.a] = audfilters(testCase.fs, testCase.Ls);
            testCase.noise_real    = randn(testCase.Ls, 1);
            testCase.noise_complex = randn(testCase.Ls, 1) + 1i*randn(testCase.Ls, 1);
        end
    end

    methods(Test)

        function testParsevalRealNoise(testCase)
            % Weighted subband energy sum_m (1/a_m)*||c_m||^2 / ||x||^2
            % should be within a loose factor for an approximate tight frame.
            c = filterbank(testCase.noise_real, testCase.g, testCase.a);

            weighted_energy = 0;
            for m = 1:length(c)
                weighted_energy = weighted_energy + ...
                    sum(abs(c{m}(:)).^2) / testCase.a(m);
            end
            input_energy = sum(abs(testCase.noise_real).^2);

            ratio = weighted_energy / input_energy;
            testCase.verifyGreaterThan(ratio, 0.01, 'Weighted energy ratio too low');
            testCase.verifyLessThan(ratio, 200,  'Weighted energy ratio too high');
        end

        function testParsevalComplexNoise(testCase)
            % Same test for complex input
            c = filterbank(testCase.noise_complex, testCase.g, testCase.a);

            weighted_energy = 0;
            for m = 1:length(c)
                weighted_energy = weighted_energy + ...
                    sum(abs(c{m}(:)).^2) / testCase.a(m);
            end
            input_energy = sum(abs(testCase.noise_complex).^2);

            ratio = weighted_energy / input_energy;
            testCase.verifyGreaterThan(ratio, 0.01);
            testCase.verifyLessThan(ratio, 200);
        end

        function testParsevalZeroInput(testCase)
            % Zero input -> zero subband energy
            c = filterbank(zeros(testCase.Ls,1), testCase.g, testCase.a);

            total_energy = sum(cellfun(@(cm) sum(abs(cm(:)).^2), c));
            testCase.verifyEqual(total_energy, 0, 'AbsTol', 1e-20);
        end

        function testParsevalLinearScaling(testCase)
            % filterbank(alpha*x) scaled by |alpha|^2 in energy
            rng(42);
            alpha = 3.7;
            c1 = filterbank(testCase.noise_real,       testCase.g, testCase.a);
            c2 = filterbank(alpha*testCase.noise_real, testCase.g, testCase.a);

            e1 = sum(cellfun(@(cm) sum(abs(cm(:)).^2), c1));
            e2 = sum(cellfun(@(cm) sum(abs(cm(:)).^2), c2));

            testCase.verifyEqual(e2, alpha^2 * e1, 'RelTol', 1e-10, ...
                'Subband energy does not scale with |alpha|^2');
        end

    end

end
