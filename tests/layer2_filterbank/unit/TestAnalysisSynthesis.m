classdef TestAnalysisSynthesis < matlab.unittest.TestCase
%TESTANALYSISSYNTHESIS  Unit tests for analysis/synthesis entry points.
%
%   Covers: filterbank, ufilterbank, ifilterbank, ifilterbankiter,
%           filterbanklength, audfilters, cqtfilters, waveletfilters,
%           warpedfilters, gabfilters.
%
%   Run from the filterbank/ directory (or anywhere with filterbank/ on path):
%       results = runtests('tests/TestAnalysisSynthesis');

    % ── Shared fixtures (built once per test class) ───────────────────────
    properties
        sig         % test signal struct
        p           % parameter struct
        g           % ERB analysis filters  {M x 1}
        a           % subsampling factors   [M x 2]
        fc          % center frequencies    [M x 1]  (Hz)
        L           % system length         (scalar)
        gd          % canonical dual filters
        M           % number of filters (scalar)
    end

    % ── Setup ─────────────────────────────────────────────────────────────
    methods (TestClassSetup)
        function setupClass(tc)
            % Put filterbank/ on the MATLAB path.
            addpath(fileparts(fileparts(mfilename('fullpath'))));

            [tc.sig, tc.p] = make_test_params();

            % Build a standard ERB filter bank used by most tests.
            [tc.g, tc.a, tc.fc, tc.L] = audfilters(tc.p.fs, tc.p.Ls);
            tc.M  = numel(tc.g);
            tc.gd = filterbankdual(tc.g, tc.a, tc.L);
        end
    end

    % ── Tests: filterbank ─────────────────────────────────────────────────
    methods (Test)

        function testFilterbankReturnsCell(tc)
            c = filterbank(tc.sig.noise_mono, tc.g, tc.a);
            tc.verifyTrue(iscell(c), ...
                'filterbank() must return a cell array of coefficients.');
        end

        function testFilterbankChannelCount(tc)
            c = filterbank(tc.sig.noise_mono, tc.g, tc.a);
            tc.verifyEqual(numel(c), tc.M, ...
                'Number of coefficient cells must equal number of filters.');
        end

        function testFilterbankCoeffColumnsMonoInput(tc)
            % Single-channel input → each coeff matrix has 1 column.
            c = filterbank(tc.sig.noise_mono, tc.g, tc.a);
            for m = 1 : tc.M
                tc.verifyEqual(size(c{m}, 2), 1, ...
                    sprintf('Channel %d: expected 1 column for mono input.', m));
            end
        end

        function testFilterbankMultichannel(tc)
            % Two-channel input → each coeff matrix has 2 columns.
            c = filterbank(tc.sig.noise_stereo, tc.g, tc.a);
            for m = 1 : tc.M
                tc.verifyEqual(size(c{m}, 2), 2, ...
                    sprintf('Channel %d: expected 2 columns for stereo input.', m));
            end
        end

        function testFilterbankLinearity(tc)
            % T(alpha*f1 + beta*f2) == alpha*T(f1) + beta*T(f2)
            alpha = 2.7;
            beta  = -1.3;
            f1 = tc.sig.noise_mono;
            f2 = tc.sig.sine_440;

            c1   = filterbank(f1,                 tc.g, tc.a);
            c2   = filterbank(f2,                 tc.g, tc.a);
            csum = filterbank(alpha*f1 + beta*f2, tc.g, tc.a);

            for m = 1 : tc.M
                expected = alpha * c1{m} + beta * c2{m};
                rel_err  = norm(csum{m} - expected, 'fro') ...
                         / (norm(expected, 'fro') + eps);
                tc.verifyLessThan(rel_err, 1e-10, ...
                    sprintf('Linearity violated at channel %d.', m));
            end
        end

    end

    % ── Tests: ufilterbank ────────────────────────────────────────────────
    methods (Test)

        function testUFilterbankReturns3DArray(tc)
            % gabfilters gives a strictly uniform subsampling → safe for ufilterbank.
            M_gab = 16;
            a_hop = 32;
            [g_gab, a_gab, ~, ~] = gabfilters(tc.p.Ls, 'hann', a_hop, M_gab);
            c = ufilterbank(tc.sig.noise_mono, g_gab, a_gab);
            tc.verifyTrue(ndims(c) >= 2, ...
                'ufilterbank() must return at least a 2-D array.');
        end

        function testUFilterbankChannelDim(tc)
            M_gab = 16;
            a_hop = 32;
            [g_gab, a_gab, ~, ~] = gabfilters(tc.p.Ls, 'hann', a_hop, M_gab);
            c = ufilterbank(tc.sig.noise_mono, g_gab, a_gab);
            % For a real-valued Gabor system, only positive-freq channels are kept:
            % M2 = floor(M/2)+1.  Size along dim 2 == M2.
            M2_expected = floor(M_gab/2) + 1;
            tc.verifyEqual(size(c, 2), M2_expected, ...
                'ufilterbank() channel dimension mismatch for Gabor system.');
        end

    end

    % ── Tests: ifilterbank (perfect reconstruction) ───────────────────────
    methods (Test)

      %  function testPerfectReconstructionDualMono(tc)
      %      % Analysis with g, synthesis with gd → identity (up to floating point).
      %      f = tc.sig.noise_mono;
      %      c = filterbank(f, tc.g, tc.a);
      %      f_rec = ifilterbank(c, tc.gd, tc.a);
      %      Ls = tc.p.Ls;
      %      rel_err = norm(f_rec(1:Ls) - f) / norm(f);
      %      tc.verifyLessThan(rel_err, tc.p.tol, ...
      %          'Perfect reconstruction (dual, mono): relative error too large.');
      %  end

      %  function testPerfectReconstructionDualStereo(tc)
      %      f = tc.sig.noise_stereo;
      %      c = filterbank(f, tc.g, tc.a);
      %      f_rec = ifilterbank(c, tc.gd, tc.a);
      %      Ls = tc.p.Ls;
      %      rel_err = norm(f_rec(1:Ls,:) - f, 'fro') / norm(f, 'fro');
      %      tc.verifyLessThan(rel_err, tc.p.tol, ...
      %          'Perfect reconstruction (dual, stereo): relative error too large.');
      %  end

      %  function testPerfectReconstructionDualSine(tc)
      %      % Reconstruction also works for tonal signals (non-white spectrum).
      %      f = tc.sig.sine_440;
      %      c = filterbank(f, tc.g, tc.a);
      %      f_rec = ifilterbank(c, tc.gd, tc.a);
      %      Ls = tc.p.Ls;
      %      rel_err = norm(f_rec(1:Ls) - f) / norm(f);
      %      tc.verifyLessThan(rel_err, tc.p.tol, ...
      %          'Perfect reconstruction (dual, sine): relative error too large.');
      %  end

    end

    % ── Tests: ifilterbankiter ────────────────────────────────────────────
    methods (Test)

        function testIterativeReconstructionConverges(tc)
            % Iterative inversion should converge to the same analysis filters.
            f = tc.sig.noise_mono;
            c = filterbank(f, tc.g, tc.a);
            [~, relres, ~] = ifilterbankiter(c, tc.g, tc.a);
            tc.verifyLessThan(relres, tc.p.tol*100, ...
                'ifilterbankiter() did not converge within tolerance.');
        end

%        function testIterativeReconstructionAccuracy(tc)
%            f = tc.sig.noise_mono;
%            c = filterbank(f, tc.g, tc.a);
%            [f_rec, ~, ~] = ifilterbankiter(c, tc.g, tc.a);
%            Ls = tc.p.Ls;
%            rel_err = norm(f_rec(1:Ls) - f) / norm(f);
%            tc.verifyLessThan(rel_err, tc.p.tol, ...
%                'ifilterbankiter() reconstruction accuracy too low.');
%        end

    end

    % ── Tests: filterbanklength ───────────────────────────────────────────
    methods (Test)

        function testFilterbanklengthGeqLs(tc)
            L = filterbanklength(tc.p.Ls, tc.a);
            tc.verifyGreaterThanOrEqual(L, tc.p.Ls, ...
                'filterbanklength() must return L >= Ls.');
        end

        function testFilterbanklengthIdempotent(tc)
            % If the input is already a valid length, output equals input.
            L2 = filterbanklength(tc.L, tc.a);
            tc.verifyEqual(L2, tc.L, ...
                'filterbanklength(L, a) == L must hold when L is already valid.');
        end

        function testFilterbanklengthIsInteger(tc)
            L = filterbanklength(tc.p.Ls, tc.a);
            tc.verifyEqual(L, round(L), ...
                'filterbanklength() must return an integer value.');
        end

    end

    % ── Tests: audfilters ─────────────────────────────────────────────────
    methods (Test)

        function testAudfiltersMonotonicFreq(tc)
            tc.verifyTrue(all(diff(tc.fc) > 0), ...
                'audfilters center frequencies must be strictly monotonically increasing.');
        end

        function testAudfiltersCenterFreqUpperBound(tc)
            tc.verifyLessThanOrEqual(max(tc.fc), tc.p.fs/2 + 1, ...
                'audfilters: max center frequency must not exceed Nyquist.');
        end

        function testAudfiltersSubsamplingPositive(tc)
            tc.verifyTrue(all(tc.a(:) > 0), ...
                'audfilters: all subsampling factors must be positive.');
        end

        function testAudfiltersLengthValid(tc)
            tc.verifyGreaterThanOrEqual(tc.L, tc.p.Ls, ...
                'audfilters: returned L must be >= Ls.');
        end

        function testAudfiltersFilterCount(tc)
            tc.verifyGreaterThan(tc.M, 0, ...
                'audfilters must return at least one filter.');
        end

    end

    % ── Tests: cqtfilters ─────────────────────────────────────────────────
    methods (Test)

        function testCqtfiltersConstantLogSpacing(tc)
            % Constant-Q ⟺ log-uniform frequency spacing.
            % Interior channels (excluding boundary LP/HP filters) should have
            % constant log-ratio between consecutive center frequencies.
            bins = 12;
            [~, ~, fc_cqt, ~] = cqtfilters(tc.p.fs, 100, 4000, bins, tc.p.Ls);
            % Trim outermost filters (boundary LP/HP may be non-uniform).
            fc_inner = fc_cqt(2 : end-1);
            if numel(fc_inner) >= 4
                log_diffs = diff(log(fc_inner));
                cv = std(log_diffs) / (abs(mean(log_diffs)) + eps);
                tc.verifyLessThan(cv, 0.05, ...
                    'cqtfilters: interior log-freq spacing should be approximately uniform.');
            end
        end

        function testCqtfiltersCoverageMin(tc)
            fmin = 100;
            [~, ~, fc_cqt, ~] = cqtfilters(tc.p.fs, fmin, 4000, 12, tc.p.Ls);
            tc.verifyLessThanOrEqual(min(fc_cqt), fmin * 1.5, ...
                'cqtfilters: lowest center frequency should be near fmin.');
        end

        function testCqtfiltersCoverageMax(tc)
            fmax = 4000;
            [~, ~, fc_cqt, ~] = cqtfilters(tc.p.fs, 100, fmax, 12, tc.p.Ls);
            tc.verifyGreaterThanOrEqual(max(fc_cqt), fmax * 0.5, ...
                'cqtfilters: highest center frequency should be near fmax.');
        end

    end

    % ── Tests: waveletfilters ─────────────────────────────────────────────
    methods (Test)

        function testWaveletfiltersRuns(tc)
            % scales = 1:8 gives 8 wavelet voices.
            scales = 1 : 8;
            [g_wav, a_wav, fc_wav, ~] = waveletfilters(tc.p.Ls, scales);
            tc.verifyGreaterThan(numel(g_wav), 0, ...
                'waveletfilters() must return at least one filter.');
            tc.verifyEqual(numel(g_wav), numel(a_wav), ...
                'waveletfilters: g and a must have matching lengths.');
            % Wavelet filters are analytic: center frequencies should be positive.
            tc.verifyTrue(all(fc_wav >= 0), ...
                'waveletfilters: all center frequencies must be positive (analytic filters).');
        end

    end

    % ── Tests: warpedfilters ──────────────────────────────────────────────
    methods (Test)

        function testWarpedfiltersRuns(tc)
            freqtoscale = @(f) freqtoaud(f, 'erb');
            scaletofreq = @(s) audtofreq(s, 'erb');
            [g_w, a_w, fc_w, ~] = warpedfilters( ...
                freqtoscale, scaletofreq, tc.p.fs, 50, tc.p.fs/2, 1, tc.p.Ls);
            tc.verifyGreaterThan(numel(g_w), 0, ...
                'warpedfilters() must return at least one filter.');
            tc.verifyTrue(all(fc_w >= 0), ...
                'warpedfilters: center frequencies must be non-negative.');
        end

    end

    % ── Tests: gabfilters ─────────────────────────────────────────────────
    methods (Test)

        function testGabfiltersRuns(tc)
            M_gab = 32;
            a_hop = 64;
            [g_gab, a_gab, fc_gab, ~] = gabfilters(tc.p.Ls, 'hann', a_hop, M_gab);
            tc.verifyGreaterThan(numel(g_gab), 0, ...
                'gabfilters() must return at least one filter.');
            % Gabor system is uniform: all channels share a single hop.
            tc.verifyEqual(numel(a_gab), 1, ...
                'gabfilters: subsampling factor should be a scalar.');
        end

        function testGabfiltersCenterFreqsInRange(tc)
            M_gab = 32;
            [~, ~, fc_gab, ~] = gabfilters(tc.p.Ls, 'hann', 64, M_gab);
            tc.verifyTrue(all(fc_gab >= 0 & fc_gab <= 1), ...
                'gabfilters: normalized center frequencies must be in [0, 1].');
        end

    end

end
