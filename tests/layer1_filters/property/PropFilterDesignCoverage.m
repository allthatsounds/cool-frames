classdef PropFilterDesignCoverage < matlab.unittest.TestCase
%PROPFILTERDESIGNCOVERAGE  Frame-theoretic coverage properties for filter design functions.
%
%   Tests applied to audfilters, cqtfilters, and gabfilters:
%
%   1. Output consistency: length(g) == length(a) == length(fc)
%   2. Frequency coverage: no dead bins — sum_m |H_m(k)|^2 > 0 for all k
%   3. Frame lower bound: sum_m |H_m(k)|^2 * (L/a_m) >= A > 0 for all k
%   4. Monotone centre frequencies: fc(1) < fc(2) < ... < fc(M)
%   5. Subsampling compatibility: all a(m) divide L exactly ('regsampling' mode)

    properties
        fs = 8000
        Ls = 1024
        tol = 1e-6
    end

    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
        end
    end

    % ── Helper: compute weighted frame response ───────────────────────────────
    methods (Access = private)

        function R = frameResponse(tc, g, a, L)
            % R(k) = sum_m |H_m(k)|^2 * (L / a_m)
            R = zeros(L, 1);
            for m = 1:numel(g)
                H    = comp_transferfunction(g{m}, L);
                am   = a(m, 1);
                R    = R + abs(H).^2 * (L / am);
            end
        end

    end

    % ── audfilters: output consistency ───────────────────────────────────────
    methods (Test)

        function testAudfiltersOutputConsistency(tc)
            [g, a, fc, L] = audfilters(tc.fs, tc.Ls);
            tc.verifyEqual(numel(g), numel(fc), ...
                'audfilters: length(g) must equal length(fc)');
            tc.verifyEqual(size(a, 1), numel(g), ...
                'audfilters: length(a) must equal length(g)');
            tc.verifyGreaterThan(numel(g), 0, 'audfilters: must return at least one filter');
            tc.verifyGreaterThan(L, 0, 'audfilters: returned L must be positive');
        end

        function testAudfiltersMonotoneCentreFreqs(tc)
            [~, ~, fc] = audfilters(tc.fs, tc.Ls);
            tc.verifyTrue(all(diff(fc) > 0), ...
                'audfilters: centre frequencies must be strictly increasing');
        end

        function testAudfiltersSubsamplingCompatibility(tc)
            [g, a, ~, L] = audfilters(tc.fs, tc.Ls, 'regsampling');
            for m = 1:numel(g)
                am = a(m, 1);
                tc.verifyEqual(mod(L, am), 0, ...
                    sprintf('audfilters: a(%d)=%d does not divide L=%d', m, am, L));
            end
        end

        function testAudfiltersCoverageNoBinDead(tc)
            % Use 'complex' mode so filters cover ALL L DFT bins.
            % Real mode (default) constructs only positive-frequency filters;
            % the negative-frequency half of the DFT remains uncovered and
            % min(R) would be 0 for those bins.
            [g, ~, ~, L] = audfilters(tc.fs, tc.Ls, 'complex');
            R = zeros(L, 1);
            for m = 1:numel(g)
                H = comp_transferfunction(g{m}, L);
                R = R + abs(H).^2;
            end
            tc.verifyGreaterThan(min(R), 0, ...
                'audfilters (complex): every DFT bin must have non-zero total energy');
        end

        function testAudfiltersFrameLowerBound(tc)
            % Use 'complex' mode and the recommended L.
            % Real mode leaves negative-frequency bins with R=0 (no filters
            % cover them), so the frame lower bound is 0 in real mode.
            [g, a, ~, L] = audfilters(tc.fs, tc.Ls, 'complex');
            R = tc.frameResponse(g, a, L);
            tc.verifyGreaterThan(min(R), 0, ...
                'audfilters (complex): frame lower bound must be positive');
        end

    end

    % ── cqtfilters: output consistency ───────────────────────────────────────
    methods (Test)

        function testCqtfiltersOutputConsistency(tc)
            fmin = 50;  fmax = tc.fs / 2;
            [g, a, fc] = cqtfilters(tc.fs, fmin, fmax, 12, tc.Ls);
            tc.verifyEqual(numel(g), numel(fc), ...
                'cqtfilters: length(g) must equal length(fc)');
            tc.verifyEqual(size(a, 1), numel(g), ...
                'cqtfilters: length(a) must equal length(g)');
        end

        function testCqtfiltersMonotoneCentreFreqs(tc)
            fmin = 50;  fmax = tc.fs / 2;
            [~, ~, fc] = cqtfilters(tc.fs, fmin, fmax, 12, tc.Ls);
            tc.verifyTrue(all(diff(fc) > 0), ...
                'cqtfilters: centre frequencies must be strictly increasing');
        end

        function testCqtfiltersConstantQ(tc)
            % For a constant-Q filterbank, fc(m)/bandwidth(m) is approximately constant.
            % We approximate bandwidth via the -3dB width in DFT bins.
            fmin = 100;  fmax = tc.fs / 2;
            bins_per_oct = 12;
            [g, ~, fc] = cqtfilters(tc.fs, fmin, fmax, bins_per_oct, tc.Ls);
            L = tc.Ls;
            Q_vals = zeros(numel(g), 1);
            for m = 1:numel(g)
                H   = abs(comp_transferfunction(g{m}, L));
                bw  = sum(H > max(H) * 0.5);   % half-power width in bins
                if bw > 0
                    fc_bin   = fc(m) / (tc.fs/2) * L/2;
                    Q_vals(m) = fc_bin / bw;
                end
            end
            % Exclude the two boundary filters (DC and Nyquist)
            Q_inner = Q_vals(2:end-1);
            Q_inner = Q_inner(Q_inner > 0);
            if numel(Q_inner) > 2
                cv = std(Q_inner) / mean(Q_inner);   % coefficient of variation
                tc.verifyLessThan(cv, 0.5, ...
                    'cqtfilters: Q factor should be approximately constant across channels');
            end
        end

    end

    % ── gabfilters: output consistency ───────────────────────────────────────
    methods (Test)

        function testGabfiltersOutputConsistency(tc)
            % gabfilters(Ls, g, a, M): g=window, a=hop, M=channels.
            % The output 'a' is the SCALAR hop factor (same as the input
            % a_hop), NOT a per-filter vector.  Use numel(fc)==numel(g)
            % to verify that the filter count matches the centre-frequency
            % count.  Real mode gives floor(M/2)+1 filters.
            a_hop = 64;  M_ch = 128;
            [g, a, fc, L] = gabfilters(tc.Ls, 'hann', a_hop, M_ch);
            tc.verifyGreaterThan(numel(g), 0, 'gabfilters: must return at least one filter');
            tc.verifyEqual(numel(fc), numel(g), ...
                'gabfilters: numel(fc) must equal numel(g)');
            % Output a is the scalar hop factor (unchanged from input)
            tc.verifyEqual(a, a_hop, ...
                'gabfilters: output a must equal the input hop factor');
            tc.verifyGreaterThan(L, 0, 'gabfilters: recommended L must be positive');
        end

        function testGabfiltersCoverageNoBinDead(tc)
            % Use 'complex' mode so that filters cover ALL DFT bins.
            % Real mode (default) only creates floor(M/2)+1 positive-
            % frequency filters, leaving negative-frequency bins uncovered.
            a_hop = 64;  M_ch = 128;
            [g, ~, ~, L] = gabfilters(tc.Ls, 'hann', a_hop, M_ch, 'complex');
            R = zeros(L, 1);
            for m = 1:numel(g)
                H = comp_transferfunction(g{m}, L);
                R = R + abs(H).^2;
            end
            tc.verifyGreaterThan(min(R), 0, ...
                'gabfilters (complex): every DFT bin must have non-zero total energy');
        end

    end

end
