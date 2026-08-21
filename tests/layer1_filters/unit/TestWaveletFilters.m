classdef TestWaveletFilters < matlab.unittest.TestCase
% TestWaveletFilters   Unit tests for waveletfilters and freqwavelet.
%
%   Tests:
%     1. Return-value structure
%     2. Wavelet types (cauchy, morse, morlet, fbsp, analyticsp, cplxsp)
%     3. Sampling modes (regsampling, uniform, fractional, fractionaluniform)
%     4. Lowpass modes (single, repeat, none)
%     5. Frequency ranges (real, complex, analytic)
%     6. Frame bounds / reconstruction
%     7. freqwavelet output formats

    properties (Constant)
        Ls = 4096;
        scales = linspace(10, 0.1, 50);
    end

    methods (Test)

        %% 1. Return-value structure
        function testReturnStructure(testCase)
            [g, a, fc, L, info] = waveletfilters(testCase.Ls, testCase.scales);
            testCase.verifyTrue(iscell(g));
            testCase.verifyGreaterThan(numel(g), 0);
            testCase.verifyGreaterThanOrEqual(L, testCase.Ls);
            testCase.verifyTrue(isstruct(info));
            testCase.verifyEqual(numel(fc), numel(g));
        end

        function testFilterDescriptorFields(testCase)
            [g, ~, ~, ~, ~] = waveletfilters(testCase.Ls, testCase.scales);
            for k = 1:numel(g)
                testCase.verifyTrue(isfield(g{k}, 'H') || isfield(g{k}, 'h'));
                testCase.verifyTrue(isfield(g{k}, 'foff'));
            end
        end

        %% 2. Wavelet types
        function testCauchyWavelet(testCase)
            [g, a, ~, L] = waveletfilters(testCase.Ls, testCase.scales, {'cauchy', 300});
            testCase.verifyGreaterThan(numel(g), 0);
        end

        function testMorseWavelet(testCase)
            [g, a, ~, L] = waveletfilters(testCase.Ls, testCase.scales, {'morse', 100, 0, 3});
            testCase.verifyGreaterThan(numel(g), 0);
        end

        function testMorletWavelet(testCase)
            [g, a, ~, L] = waveletfilters(testCase.Ls, testCase.scales, {'morlet', 6});
            testCase.verifyGreaterThan(numel(g), 0);
        end

        function testFbspWavelet(testCase)
            [g, a, ~, L] = waveletfilters(testCase.Ls, testCase.scales, {'fbsp', 4, 3});
            testCase.verifyGreaterThan(numel(g), 0);
        end

        function testAnalyticspWavelet(testCase)
            [g, a, ~, L] = waveletfilters(testCase.Ls, testCase.scales, {'analyticsp', 3, 2});
            testCase.verifyGreaterThan(numel(g), 0);
        end

        function testCplxspWavelet(testCase)
            [g, a, ~, L] = waveletfilters(testCase.Ls, testCase.scales, {'cplxsp', 3, 2});
            testCase.verifyGreaterThan(numel(g), 0);
        end

        %% 3. Sampling modes
        function testRegsampling(testCase)
            [g, a, ~, L] = waveletfilters(testCase.Ls, testCase.scales, 'regsampling');
            testCase.verifyGreaterThanOrEqual(L, testCase.Ls);
            testCase.verifyTrue(all(a(:,end) == 1) || size(a,2) == 1);
        end

        function testUniformSampling(testCase)
            [g, a, ~, L] = waveletfilters(testCase.Ls, testCase.scales, 'uniform');
            if size(a,2) == 1
                testCase.verifyTrue(all(a == a(1)));
            end
        end

        function testFractionalSampling(testCase)
            [g, a, ~, L] = waveletfilters(testCase.Ls, testCase.scales, 'fractional');
            testCase.verifyEqual(L, testCase.Ls);
            testCase.verifyEqual(size(a, 2), 2);
        end

        %% 4. Lowpass modes
        function testSingleLowpass(testCase)
            [g, ~, fc, ~, info] = waveletfilters(testCase.Ls, testCase.scales, 'single');
            testCase.verifyEqual(fc(1), 0);
        end

        function testRepeatLowpass(testCase)
            [g, ~, fc, ~] = waveletfilters(testCase.Ls, testCase.scales, 'repeat');
            testCase.verifyGreaterThan(numel(g), numel(testCase.scales));
        end

        function testNoneLowpass(testCase)
            [g, ~, fc, ~] = waveletfilters(testCase.Ls, testCase.scales, 'none');
            testCase.verifyEqual(numel(g), numel(testCase.scales));
        end

        %% 5. Frequency ranges
        function testComplexRange(testCase)
            [g, ~, fc, ~] = waveletfilters(testCase.Ls, testCase.scales, 'complex');
            testCase.verifyTrue(any(fc < 0));
        end

        %% 6. Reconstruction test (uniform sampling for simplicity)
        function testReconstructionCauchy(testCase)
            Ls = 512;
            scales = linspace(5, 0.2, 20);
            [g, a, ~, L] = waveletfilters(Ls, scales, {'cauchy', 300}, 'uniform', 'single');
            gd = filterbankrealdual(g, a, L);
            f = randn(L, 1);
            c = filterbank(f, g, a);
            r = 2 * real(ifilterbank(c, gd, a));
            testCase.verifyLessThan(norm(r(1:Ls) - f(1:Ls)) / norm(f(1:Ls)), 1e-6);
        end

        function testReconstructionFbsp(testCase)
            Ls = 512;
            scales = linspace(5, 0.2, 20);
            [g, a, ~, L] = waveletfilters(Ls, scales, {'fbsp', 4, 3}, 'uniform', 'single');
            gd = filterbankrealdual(g, a, L);
            f = randn(L, 1);
            c = filterbank(f, g, a);
            r = 2 * real(ifilterbank(c, gd, a));
            testCase.verifyLessThan(norm(r(1:Ls) - f(1:Ls)) / norm(f(1:Ls)), 1e-6);
        end

        %% 7. freqwavelet output formats
        function testFreqwaveletFull(testCase)
            L = 1024;
            [H, info] = freqwavelet({'cauchy', 300}, L, 1);
            testCase.verifyEqual(size(H, 1), L);
        end

        function testFreqwaveletEcon(testCase)
            L = 1024;
            [H, info] = freqwavelet({'cauchy', 300}, L, [1 2 4], 'econ');
            testCase.verifyTrue(iscell(H));
            testCase.verifyEqual(numel(H), 3);
        end

        function testFreqwaveletAsfreqfilter(testCase)
            L = 1024;
            [H, info] = freqwavelet({'cauchy', 300}, L, [1 2 4], 'asfreqfilter');
            testCase.verifyTrue(iscell(H));
            for k = 1:numel(H)
                testCase.verifyTrue(isstruct(H{k}));
                testCase.verifyTrue(isfield(H{k}, 'H'));
                testCase.verifyTrue(isfield(H{k}, 'foff'));
            end
        end

        %% 8. Wavelet peak position
        function testWaveletPeakPosition(testCase)
            L = 4096;
            [H, info] = freqwavelet({'cauchy', 300}, L, 1, 'full');
            [~, peak_idx] = max(abs(H));
            expected_bin = round(info.fc * L / 2);
            testCase.verifyLessThan(abs(peak_idx - expected_bin - 1), 3);
        end

    end
end
