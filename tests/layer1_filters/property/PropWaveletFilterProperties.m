classdef PropWaveletFilterProperties < matlab.unittest.TestCase
% PropWaveletFilterProperties   Property tests for waveletfilters.
%
%   Properties:
%     P1. Wavelet peak at expected centre frequency
%     P2. Support monotonically decreasing with scale
%     P3. filterbanklength divides L for regsampling
%     P4. Positive frame response for all tested configurations

    methods (Test)

        %% P1. Wavelet peak at expected centre frequency
        function testPeakAtCenterFrequency(testCase)
            L = 4096;
            scales = [1, 2, 4, 8];
            for s = scales
                [H, info] = freqwavelet({'cauchy', 300}, L, s, 'full');
                [~, peak_idx] = max(abs(H));
                expected_bin = round(info.fc * L / 2) + 1; % MATLAB 1-based
                testCase.verifyLessThan(abs(peak_idx - expected_bin), 3, ...
                    sprintf('Peak mismatch for scale=%g', s));
            end
        end

        %% P2. Support monotonically decreasing with scale
        function testSupportDecreasesWithScale(testCase)
            L = 8192;
            scales = [0.2, 0.5, 1, 2, 5, 10];
            prev_fsupp = Inf;
            for s = scales
                [~, info] = freqwavelet({'cauchy', 300}, L, s, 'full');
                cur_fsupp = info.fsupp;
                testCase.verifyLessThanOrEqual(cur_fsupp, prev_fsupp + 2);
                prev_fsupp = cur_fsupp;
            end
        end

        %% P3. filterbanklength divides L for regsampling
        function testFilterbanklengthDivision(testCase)
            Ls = 4096;
            scales = linspace(10, 0.1, 40);
            [~, a, ~, L] = waveletfilters(Ls, scales, 'regsampling');
            for k = 1:numel(a)
                testCase.verifyEqual(mod(L, a(k)), 0);
            end
        end

        %% P4. Positive frame response
        function testPositiveFrameResponse(testCase)
            Ls = 4096;
            scales = linspace(10, 0.1, 40);
            types = {{'cauchy', 300}, {'morlet', 6}, {'fbsp', 4, 3}};
            for t = 1:numel(types)
                [g, a, ~, L] = waveletfilters(Ls, scales, types{t}, 'uniform', 'single');
                resp = filterbankresponse(g, a, L, 'real');
                testCase.verifyGreaterThan(min(resp), 0, ...
                    sprintf('Frame response has zeros for type %d', t));
            end
        end

    end
end
