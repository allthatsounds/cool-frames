classdef PropLinearity < matlab.unittest.TestCase
    % PropLinearity: property tests for linearity of filterbank analysis
    % Property: filterbank(alpha*x1 + x2, g, a){m} = alpha*filterbank(x1,g,a){m}
    %                                                 + filterbank(x2,g,a){m}
    %
    % Uses the public filterbank() API, which internally routes to
    % comp_filterbank_fft / comp_filterbank_fftbl / comp_filterbank_td.

    properties
        Ls
        fs
        g    % filter structs from audfilters
        a    % subsampling factors
    end

    methods(TestClassSetup)
        function setupTestClass(testCase)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), ...
                                 '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
            testCase.Ls = 1024;
            testCase.fs = 8000;
            [testCase.g, testCase.a] = audfilters(testCase.fs, testCase.Ls);
        end
    end

    methods(Test)

        function testLinearityAnalysisPath(testCase)
            % filterbank(alpha*x1 + x2) = alpha*filterbank(x1) + filterbank(x2)
            rng(42);
            num_trials = 50;

            for trial = 1:num_trials
                x1    = randn(testCase.Ls, 1) + 1i*randn(testCase.Ls, 1);
                x2    = randn(testCase.Ls, 1) + 1i*randn(testCase.Ls, 1);
                alpha = randn() + 1i*randn();

                c_comb = filterbank(alpha*x1 + x2, testCase.g, testCase.a);
                c1     = filterbank(x1,             testCase.g, testCase.a);
                c2     = filterbank(x2,             testCase.g, testCase.a);

                scale = abs(alpha)*norm(x1) + norm(x2);
                tol   = 1e-10 * max(scale, 1);

                for m = 1:length(c1)
                    expected = alpha * c1{m} + c2{m};
                    testCase.verifyEqual(c_comb{m}, expected, 'AbsTol', tol, ...
                        sprintf('Trial %d, subband %d: analysis linearity failed', trial, m));
                end
            end
        end

        function testLinearitySynthesisPath(testCase)
            % ifilterbank(alpha*c1 + c2) = alpha*ifilterbank(c1) + ifilterbank(c2)
            % Uses the adjoint (analysis) filters as a proxy synthesis filter.
            rng(42);
            num_trials = 50;

            % Prepare G (length-L DFT responses) for the synthesis call
            g_pre = comp_filterbank_pre(testCase.g, testCase.a, testCase.Ls);

            for trial = 1:num_trials
                alpha = randn() + 1i*randn();

                % Build random coefficients with correct sizes
                c1 = cell(length(testCase.a), 1);
                c2 = cell(length(testCase.a), 1);
                for m = 1:length(testCase.a)
                    N = ceil(testCase.Ls / testCase.a(m));
                    c1{m} = randn(N,1) + 1i*randn(N,1);
                    c2{m} = randn(N,1) + 1i*randn(N,1);
                end

                % Combined coefficients
                c_comb = cell(length(testCase.a), 1);
                for m = 1:length(testCase.a)
                    c_comb{m} = alpha*c1{m} + c2{m};
                end

                % Extract full-length G for comp_ifilterbank_fft
                isFL   = cellfun(@(x) isfield(x,'H') && numel(x.H)==testCase.Ls, g_pre);
                G_full = cellfun(@(x) x.H, g_pre(isFL), 'UniformOutput', false);
                a_full = testCase.a(isFL);
                c1_fl  = c1(isFL);
                c2_fl  = c2(isFL);
                cc_fl  = c_comb(isFL);

                if isempty(G_full), continue; end  % skip if no full-length filters

                x_comb = comp_ifilterbank_fft(cc_fl, G_full, a_full);
                x1     = comp_ifilterbank_fft(c1_fl, G_full, a_full);
                x2_r   = comp_ifilterbank_fft(c2_fl, G_full, a_full);

                expected = alpha*x1 + x2_r;
                scale    = abs(alpha)*norm(x1) + norm(x2_r);
                tol      = 1e-10 * max(scale, 1);

                testCase.verifyEqual(x_comb, expected, 'AbsTol', tol, ...
                    sprintf('Trial %d: synthesis linearity failed', trial));
            end
        end

    end

end
