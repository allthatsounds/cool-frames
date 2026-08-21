classdef TestWarpedFilters < matlab.unittest.TestCase
% TestWarpedFilters   Unit tests for warpedfilters and warpedblfilter.
%
%   Tests:
%     1. Return-value structure
%     2. Warping functions (linear, sqrt, ERB, constant-Q)
%     3. Sampling modes
%     4. Frame bounds at various redundancies
%     5. warpedblfilter individual filter construction
%     6. Reconstruction test

    properties (Constant)
        Ls = 8000;
        fs = 16000;
        fmin = 100;
    end

    methods (Test)

        %% 1. Return-value structure
        function testReturnStructure(testCase)
            fmax = testCase.fs / 2;
            [g, a, fc, L] = warpedfilters(@freqtoerb, @erbtofreq, ...
                testCase.fs, testCase.fmin, fmax, 1, testCase.Ls);
            testCase.verifyTrue(iscell(g));
            testCase.verifyGreaterThan(numel(g), 0);
            testCase.verifyEqual(numel(fc), numel(g));
            testCase.verifyGreaterThanOrEqual(L, testCase.Ls);
        end

        function testFilterDescriptorFields(testCase)
            fmax = testCase.fs / 2;
            [g, ~, ~, ~] = warpedfilters(@freqtoerb, @erbtofreq, ...
                testCase.fs, testCase.fmin, fmax, 1, testCase.Ls);
            for k = 1:numel(g)
                testCase.verifyTrue(isfield(g{k}, 'H'));
                testCase.verifyTrue(isfield(g{k}, 'foff'));
            end
        end

        %% 2. Different warping functions
        function testERBScale(testCase)
            fmax = testCase.fs / 2;
            [g, a, fc, L] = warpedfilters(@freqtoerb, @erbtofreq, ...
                testCase.fs, testCase.fmin, fmax, 1, testCase.Ls);
            testCase.verifyGreaterThan(numel(g), 5);
        end

        function testLogScale(testCase)
            warpfun = @(x) 10 * log(x);
            invfun = @(x) exp(x / 10);
            fmax = testCase.fs / 2;
            [g, a, fc, L] = warpedfilters(warpfun, invfun, ...
                testCase.fs, 50, fmax, 4, testCase.Ls);
            testCase.verifyGreaterThan(numel(g), 5);
        end

        function testLinearScale(testCase)
            warpfun = @(x) x;
            invfun = @(x) x;
            fmax = testCase.fs / 2;
            [g, a, fc, L] = warpedfilters(warpfun, invfun, ...
                testCase.fs, testCase.fmin, fmax, 1, testCase.Ls);
            testCase.verifyGreaterThan(numel(g), 0);
        end

        function testSqrtScale(testCase)
            warpfun = @(x) sqrt(x);
            invfun = @(x) x.^2;
            fmax = testCase.fs / 2;
            [g, a, fc, L] = warpedfilters(warpfun, invfun, ...
                testCase.fs, testCase.fmin, fmax, 1, testCase.Ls);
            testCase.verifyGreaterThan(numel(g), 0);
        end

        %% 3. Sampling modes
        function testRegsamplingMode(testCase)
            fmax = testCase.fs / 2;
            [g, a, ~, L] = warpedfilters(@freqtoerb, @erbtofreq, ...
                testCase.fs, testCase.fmin, fmax, 1, testCase.Ls, 'regsampling');
            testCase.verifyGreaterThanOrEqual(L, testCase.Ls);
        end

        function testFractionalMode(testCase)
            fmax = testCase.fs / 2;
            [g, a, ~, L] = warpedfilters(@freqtoerb, @erbtofreq, ...
                testCase.fs, testCase.fmin, fmax, 1, testCase.Ls, 'fractional');
            testCase.verifyEqual(L, testCase.Ls);
            testCase.verifyEqual(size(a, 2), 2);
        end

        function testUniformMode(testCase)
            fmax = testCase.fs / 2;
            [g, a, ~, L] = warpedfilters(@freqtoerb, @erbtofreq, ...
                testCase.fs, testCase.fmin, fmax, 1, testCase.Ls, 'uniform');
            if size(a, 2) == 1
                testCase.verifyTrue(all(a == a(1)));
            end
        end

        %% 4. Frame bounds at various redundancies
        function testFrameBoundsERB(testCase)
            fmax = testCase.fs / 2;
            redmuls = [1, 1.5, 2, 4, 8];
            for rm = redmuls
                [g, a, ~, L] = warpedfilters(@freqtoerb, @erbtofreq, ...
                    testCase.fs, testCase.fmin, fmax, 1, testCase.Ls, ...
                    'bwmul', 1.5, 'redmul', rm, 'real', 'fractional');
                resp = filterbankresponse(g, a, L, 'real');
                testCase.verifyGreaterThan(min(resp), 0, ...
                    sprintf('Frame response has zeros at redmul=%g', rm));
            end
        end

        %% 5. warpedblfilter individual filters
        function testWarpedBlFilter(testCase)
            g = warpedblfilter('hann', 2, 1000, testCase.fs, ...
                @freqtoerb, @erbtofreq, 'scal', 1, 'inf');
            testCase.verifyTrue(isstruct(g));
            H = g.H(testCase.Ls);
            testCase.verifyGreaterThan(numel(H), 0);
            testCase.verifyGreaterThan(max(abs(H)), 0);
        end

        %% 6. Reconstruction test
        function testReconstructionERBFractional(testCase)
            fmax = testCase.fs / 2;
            [g, a, ~, L] = warpedfilters(@freqtoerb, @erbtofreq, ...
                testCase.fs, testCase.fmin, fmax, 1, testCase.Ls, ...
                'bwmul', 1.5, 'real', 'fractional');
            gd = filterbankrealdual(g, a, L);
            f = randn(L, 1);
            c = filterbank(f, g, a);
            r = 2 * real(ifilterbank(c, gd, a));
            testCase.verifyLessThan(norm(r - f) / norm(f), 1e-6);
        end

    end
end
