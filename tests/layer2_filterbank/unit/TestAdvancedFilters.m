classdef TestAdvancedFilters < matlab.unittest.TestCase
%TESTADVANCEDFILTERS  Unit tests for custom filter construction entry points.
%
%   Covers: blfilter, freqfilter, freqwavelet, warpedblfilter, firwin,
%           freqwin, firfilter, nonu2ufilterbank, freqtoaud, audtofreq,
%           audfiltbw, audspace.

    properties
        sig
        p
        % ERB filter bank for nonu2ufilterbank tests
        g
        a
        fc
        L
        M
    end

    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
            [tc.sig, tc.p] = make_test_params();
            [tc.g, tc.a, tc.fc, tc.L] = audfilters(tc.p.fs, tc.p.Ls);
            tc.M = numel(tc.g);
        end
    end

    % ── blfilter ──────────────────────────────────────────────────────────
    methods (Test)

        function testBlfilterReturnsStruct(tc)
            g = blfilter('hann', 0.1);
            tc.verifyTrue(isstruct(g), ...
                'blfilter() must return a struct.');
        end

        function testBlfilterHasFieldH(tc)
            g = blfilter('hann', 0.1);
            tc.verifyTrue(isfield(g, 'H'), ...
                'blfilter struct must have field .H (transfer function).');
        end

        function testBlfilterFcOption(tc)
            fc_target = 0.25;
            g = blfilter('hann', 0.1, 'fc', fc_target);
            tc.verifyTrue(isstruct(g), ...
                'blfilter with fc option must return a struct.');
            % Center frequency should be stored in the struct.
            if isfield(g, 'fc')
                tc.verifyLessThan(abs(g.fc - fc_target), 0.01, ...
                    'blfilter: stored fc should match the requested value.');
            end
        end

    end

    % ── freqfilter ────────────────────────────────────────────────────────
    methods (Test)

        function testFreqfilterReturnsStruct(tc)
            H = ones(tc.L, 1);   % flat transfer function
            g = freqfilter(H, 0.0);
            tc.verifyTrue(isstruct(g), ...
                'freqfilter() must return a struct.');
        end

        function testFreqfilterHasFieldH(tc)
            H = hann(tc.L);
            g = freqfilter(H, 0.25);
            tc.verifyTrue(isfield(g, 'H'), ...
                'freqfilter struct must have field .H.');
        end

    end

    % ── freqwavelet ───────────────────────────────────────────────────────
    methods (Test)

        function testFreqwaveletCauchyReturnsStruct(tc)
            % Use 'asfreqfilter' to get a filter struct (with .H, .foff fields).
            % Default 'full' mode returns a numeric matrix, not a struct.
            g = freqwavelet('cauchy', tc.L, 'asfreqfilter');
            tc.verifyTrue(isstruct(g), ...
                'freqwavelet(''cauchy'', M, ''asfreqfilter'') must return a struct.');
        end

        function testFreqwaveletHasFieldH(tc)
            g = freqwavelet('cauchy', tc.L, 'asfreqfilter');
            tc.verifyTrue(isfield(g, 'H'), ...
                'freqwavelet struct must have field .H (frequency-domain transfer function).');
        end

    end

    % ── firwin ────────────────────────────────────────────────────────────
    methods (Test)

        function testFirwinLengthEven(tc)
            M = 32;
            g = firwin('hann', M);
            tc.verifyEqual(numel(g), M, ...
                'firwin: output length must equal requested M (even).');
        end

        function testFirwinLengthOdd(tc)
            M = 33;
            g = firwin('hann', M);
            tc.verifyEqual(numel(g), M, ...
                'firwin: output length must equal requested M (odd).');
        end

        function testFirwinSymmetry(tc)
            g = firwin('hann', 64);
            % firwin returns a DFT-ordered (whole-point even) window.
            % The symmetry condition is g(k) == g(M+2-k) for k=2..M,
            % which is equivalent to: g(2:end) == flipud(g(2:end)).
            tc.verifyLessThan(norm(g(2:end) - flipud(g(2:end))), 1e-12, ...
                'firwin(''hann'',...): window must be DFT-symmetric (g(k)==g(M+2-k)).');
        end

        function testFirwinRealValued(tc)
            g = firwin('hamming', 32);
            tc.verifyTrue(isreal(g), ...
                'firwin: output must be real-valued for standard windows.');
        end

        function testFirwinHannSumNear1(tc)
            % Hann window of length M has known sum = M/2.
            M = 64;
            g = firwin('hann', M);
            expected_sum = M / 2;
            tc.verifyLessThan(abs(sum(g) - expected_sum) / expected_sum, 0.01, ...
                'firwin(''hann'', M): window sum should be approximately M/2.');
        end

        function testFirwinVariousWindows(tc)
            % Note: firwin uses 'tria' (not 'tri') for the triangular window.
            windows = {'hamming', 'blackman', 'rect', 'tria'};
            for k = 1 : numel(windows)
                g = firwin(windows{k}, 32);
                tc.verifyEqual(numel(g), 32, ...
                    sprintf('firwin(''%s'', 32): wrong length.', windows{k}));
                tc.verifyTrue(isreal(g), ...
                    sprintf('firwin(''%s'', 32): must be real.', windows{k}));
            end
        end

    end

    % ── freqwin ───────────────────────────────────────────────────────────
    methods (Test)

        function testFreqwinLength(tc)
            M  = 64;
            bw = 0.1;
            % freqwin supports: 'gauss', 'gammatone', 'butterworth'
            g  = freqwin('gauss', M, bw);
            tc.verifyEqual(numel(g), M, ...
                'freqwin: output length must equal M.');
        end

        function testFreqwinNonNegative(tc)
            % A Gaussian frequency window is real-valued and non-negative.
            g = freqwin('gauss', 64, 0.1);
            if isreal(g)
                tc.verifyTrue(all(g(:) >= -1e-12), ...
                    'freqwin(''gauss'',...): values should be non-negative.');
            end
        end

    end

    % ── firfilter ─────────────────────────────────────────────────────────
    methods (Test)

        function testFirfilterReturnsStruct(tc)
            g = firfilter('hann', 32);
            tc.verifyTrue(isstruct(g), ...
                'firfilter() must return a struct.');
        end

        function testFirfilterHasFieldH(tc)
            g = firfilter('hann', 32);
            tc.verifyTrue(isfield(g, 'h'), ...
                'firfilter struct must have field .h (time-domain FIR coefficients).');
        end

        function testFirfilterVectorMReturnsCell(tc)
            lengths = [16, 32, 64];
            g = firfilter('hann', lengths);
            tc.verifyTrue(iscell(g), ...
                'firfilter with vector M must return a cell array.');
            tc.verifyEqual(numel(g), numel(lengths), ...
                'firfilter: cell length must match number of requested lengths.');
        end

    end

    % ── nonu2ufilterbank ──────────────────────────────────────────────────
    methods (Test)

        function testNonu2uReturnsCell(tc)
            [gu, ~, ~] = nonu2ufilterbank(tc.g, tc.a);
            tc.verifyTrue(iscell(gu), ...
                'nonu2ufilterbank: gu must be a cell array.');
        end

        function testNonu2uUniformHopIsScalar(tc)
            [~, au, ~] = nonu2ufilterbank(tc.g, tc.a);
            tc.verifyEqual(numel(au), 1, ...
                'nonu2ufilterbank: au must be a scalar (uniform hop size).');
        end

        function testNonu2uPSumsToFilterCount(tc)
            [gu, ~, p] = nonu2ufilterbank(tc.g, tc.a);
            tc.verifyEqual(sum(p), numel(gu), ...
                'nonu2ufilterbank: sum(p) must equal the number of uniform filters.');
        end

        function testNonu2uFrameBoundsEquivalence(tc)
            % The uniform equivalent bank should have the same frame bounds.
            %
            % NOTE: filterbankbounds returns AF=0 for one-sided (analytic) ERB banks
            % because negative-frequency bins are uncovered.  Using AF1 + eps as the
            % denominator with AF1=0 amplifies any floating-point difference to O(1).
            % We therefore compare effective positive-frequency bounds from
            % filterbankresponse instead of filterbankbounds.
            [gu, au, ~] = nonu2ufilterbank(tc.g, tc.a);

            L_half = floor(tc.L/2) + 1;
            gf1 = real(filterbankresponse(tc.g, tc.a,  tc.L));
            gf2 = real(filterbankresponse(gu,   au,    tc.L));

            AF1_eff = min(gf1(1:L_half));
            BF1_eff = max(gf1(1:L_half));
            AF2_eff = min(gf2(1:L_half));
            BF2_eff = max(gf2(1:L_half));

            tc.verifyLessThan(abs(AF2_eff - AF1_eff) / (AF1_eff + eps), 1e-6, ...
                'nonu2ufilterbank: lower frame bound must match original.');
            tc.verifyLessThan(abs(BF2_eff - BF1_eff) / (BF1_eff + eps), 1e-6, ...
                'nonu2ufilterbank: upper frame bound must match original.');
        end

    end

    % ── freqtoaud / audtofreq ─────────────────────────────────────────────
    methods (Test)

        function testFreqtoaudErbInvertibility(tc)
            freqs = [100, 500, 1000, 4000];
            freqs_roundtrip = audtofreq(freqtoaud(freqs, 'erb'), 'erb');
            rel_err = norm(freqs_roundtrip - freqs) / norm(freqs);
            tc.verifyLessThan(rel_err, 1e-10, ...
                'freqtoaud/audtofreq: ERB roundtrip must be identity.');
        end

        function testFreqtoaudBarkInvertibility(tc)
            freqs = [100, 500, 1000, 4000];
            freqs_rt = audtofreq(freqtoaud(freqs, 'bark'), 'bark');
            rel_err = norm(freqs_rt - freqs) / norm(freqs);
            tc.verifyLessThan(rel_err, 1e-10, ...
                'freqtoaud/audtofreq: Bark roundtrip must be identity.');
        end

        function testFreqtoaudMelInvertibility(tc)
            freqs = [100, 500, 1000, 4000];
            freqs_rt = audtofreq(freqtoaud(freqs, 'mel'), 'mel');
            rel_err = norm(freqs_rt - freqs) / norm(freqs);
            tc.verifyLessThan(rel_err, 1e-10, ...
                'freqtoaud/audtofreq: Mel roundtrip must be identity.');
        end

        function testFreqtoaudMonotonic(tc)
            freqs = 100 : 100 : 4000;
            for scale_name = {'erb', 'bark', 'mel'}
                aud_vals = freqtoaud(freqs, scale_name{1});
                tc.verifyTrue(all(diff(aud_vals) > 0), ...
                    sprintf('freqtoaud(''%s''): must be monotonically increasing.', ...
                            scale_name{1}));
            end
        end

        function testFreqtoaudShapePreserved(tc)
            freqs = [100, 500, 1000];
            aud = freqtoaud(freqs, 'erb');
            tc.verifyEqual(size(aud), size(freqs), ...
                'freqtoaud: output shape must match input shape.');
        end

    end

    % ── audfiltbw ─────────────────────────────────────────────────────────
    methods (Test)

        function testAudfiltbwPositive(tc)
            fc_test = [100, 500, 1000, 2000, 4000];
            bw = audfiltbw(fc_test, 'erb');
            tc.verifyTrue(all(bw > 0), ...
                'audfiltbw: all bandwidths must be positive.');
        end

        function testAudfiltbwMonotonicErb(tc)
            % ERB bandwidth grows monotonically with center frequency.
            fc_test = 100 : 100 : 4000;
            bw = audfiltbw(fc_test, 'erb');
            tc.verifyTrue(all(diff(bw) > 0), ...
                'audfiltbw(''erb''): bandwidth must increase monotonically with fc.');
        end

        function testAudfiltbwShapePreserved(tc)
            fc_test = [100, 500, 1000];
            bw = audfiltbw(fc_test, 'erb');
            tc.verifyEqual(size(bw), size(fc_test), ...
                'audfiltbw: output shape must match input shape.');
        end

    end

    % ── audspace ──────────────────────────────────────────────────────────
    methods (Test)

        function testAudspaceOutputLength(tc)
            N = 20;
            [fc_out, bw_out] = audspace(100, 4000, N, 'erb');
            tc.verifyEqual(numel(fc_out), N, ...
                'audspace: fc output must have exactly N elements.');
            tc.verifyEqual(numel(bw_out), N, ...
                'audspace: bw output must have exactly N elements.');
        end

        function testAudspaceBandwidthsPositive(tc)
            [~, bw_out] = audspace(100, 4000, 20, 'erb');
            tc.verifyTrue(all(bw_out > 0), ...
                'audspace: all bandwidths must be positive.');
        end

        function testAudspaceCoverageApproximate(tc)
            % Endpoints should lie within the specified frequency range.
            flo = 100;  fhi = 4000;
            [fc_out, ~] = audspace(flo, fhi, 20, 'erb');
            tc.verifyGreaterThanOrEqual(min(fc_out), flo * 0.9, ...
                'audspace: lowest center freq should be near flow.');
            tc.verifyLessThanOrEqual(max(fc_out), fhi * 1.1, ...
                'audspace: highest center freq should be near fhigh.');
        end

        function testAudspaceMonotonic(tc)
            [fc_out, ~] = audspace(100, 4000, 20, 'erb');
            tc.verifyTrue(all(diff(fc_out) > 0), ...
                'audspace: center frequencies must be strictly monotonically increasing.');
        end

    end

    % ── biquadfilter ──────────────────────────────────────────────────────
    methods (Test)

        function testBiquadfilterReturnsStruct(tc)
            g = biquadfilter(0.25, 0.05);
            tc.verifyTrue(isstruct(g), ...
                'biquadfilter() must return a struct.');
        end

        function testBiquadfilterHasRequiredFields(tc)
            g = biquadfilter(0.25, 0.05);
            tc.verifyTrue(isfield(g, 'H'),     'biquadfilter: missing field .H');
            tc.verifyTrue(isfield(g, 'foff'),  'biquadfilter: missing field .foff');
            tc.verifyTrue(isfield(g, 'r'),     'biquadfilter: missing field .r');
            tc.verifyTrue(isfield(g, 'theta'), 'biquadfilter: missing field .theta');
        end

        function testBiquadfilterStabilityConstraint(tc)
            % Pole radius must be in (0,1) for a stable IIR filter.
            g = biquadfilter(0.30, 0.04);
            tc.verifyGreaterThan(g.r, 0, ...
                'biquadfilter: pole radius r must be positive.');
            tc.verifyLessThan(g.r, 1, ...
                'biquadfilter: pole radius r must be < 1 (stability).');
        end

        function testBiquadfilterMLParametrizationFinite(tc)
            % Unconstrained ML parameters rho and phi must be finite scalars.
            g = biquadfilter(0.25, 0.05);
            tc.verifyTrue(isfield(g, 'rho') && isscalar(g.rho) && isfinite(g.rho), ...
                'biquadfilter: rho must be a finite scalar.');
            tc.verifyTrue(isfield(g, 'phi') && isscalar(g.phi) && isfinite(g.phi), ...
                'biquadfilter: phi must be a finite scalar.');
        end

        function testBiquadfilterMLRoundtrip(tc)
            % Passing the stored (rho, phi) back as overrides must reproduce the
            % same pole parameters.
            g1 = biquadfilter(0.30, 0.06);
            g2 = biquadfilter(0.30, 0.06, 'rho', g1.rho, 'phi', g1.phi);
            tc.verifyEqual(g2.r,     g1.r,     'AbsTol', 1e-10, ...
                'biquadfilter: rho/phi overrides must reproduce the same r.');
            tc.verifyEqual(g2.theta, g1.theta, 'AbsTol', 1e-10, ...
                'biquadfilter: rho/phi overrides must reproduce the same theta.');
        end

        function testBiquadfilterVectorInputReturnsCell(tc)
            fcs = [0.10, 0.30, 0.50];
            bws = 0.05 * ones(1, 3);
            gs  = biquadfilter(fcs, bws);
            tc.verifyTrue(iscell(gs), ...
                'biquadfilter with vector fc must return a cell array.');
            tc.verifyEqual(numel(gs), numel(fcs), ...
                'biquadfilter: cell length must match the number of requested filters.');
        end

        function testBiquadfilterIntegratesWithFilterbank(tc)
            % biquadfilter structs must be accepted by filterbank() without error.
            fcs   = [0.15, 0.40, 0.65];
            gs    = biquadfilter(fcs, 0.06 * ones(size(fcs)));
            a_biq = ones(numel(fcs), 1);
            c     = filterbank(tc.sig.noise_mono, gs, a_biq);
            tc.verifyEqual(numel(c), numel(fcs), ...
                'filterbank + biquadfilter: coefficient count must equal filter count.');
            for m = 1 : numel(fcs)
                tc.verifyEqual(size(c{m}, 1), tc.p.Ls, ...
                    sprintf('biquadfilter channel %d: coefficient length must equal Ls.', m));
            end
        end

        function testBiquadfilterFsOptionConsistentWithNormalized(tc)
            % With 'fs', fc and bw are in Hz; the pole angle must equal
            % the one obtained from the equivalent normalized fc.
            fs_hz  = 8000;
            fc_hz  = 1000;
            bw_hz  = 200;
            g_hz   = biquadfilter(fc_hz, bw_hz, 'fs', fs_hz);
            fc_n   = fc_hz / fs_hz * 2;   % LTFAT normalized (1 = Nyquist)
            g_n    = biquadfilter(fc_n, bw_hz / fs_hz * 2);
            tc.verifyEqual(g_hz.theta, g_n.theta, 'AbsTol', 1e-10, ...
                'biquadfilter: Hz and normalized inputs must give the same pole angle.');
        end

    end

    % ── warpedblfilter ────────────────────────────────────────────────────
    methods (Test)

        function testWarpedblfilterReturnsStruct(tc)
            f2s = @(f) freqtoaud(f, 'erb');
            s2f = @(s) audtofreq(s, 'erb');
            g   = warpedblfilter('hann', 1.0, 500, tc.p.fs, f2s, s2f);
            tc.verifyTrue(isstruct(g), ...
                'warpedblfilter() must return a struct.');
        end

        function testWarpedblfilterHasFieldH(tc)
            f2s = @(f) freqtoaud(f, 'erb');
            s2f = @(s) audtofreq(s, 'erb');
            g   = warpedblfilter('hann', 1.0, 500, tc.p.fs, f2s, s2f);
            tc.verifyTrue(isfield(g, 'H'), ...
                'warpedblfilter struct must have field .H.');
        end

        function testWarpedblfilterVectorFcReturnsCell(tc)
            f2s = @(f) freqtoaud(f, 'erb');
            s2f = @(s) audtofreq(s, 'erb');
            fcs = [200, 500, 1000, 2000];
            gs  = warpedblfilter('hann', 1.0, fcs, tc.p.fs, f2s, s2f);
            tc.verifyTrue(iscell(gs), ...
                'warpedblfilter with vector fc must return a cell array.');
            tc.verifyEqual(numel(gs), numel(fcs), ...
                'warpedblfilter: cell length must match input vector length.');
        end

        function testWarpedblfilterIntegratesWithFilterbank(tc)
            f2s = @(f) freqtoaud(f, 'erb');
            s2f = @(s) audtofreq(s, 'erb');
            fcs = [300, 800, 1600];
            gs  = warpedblfilter('hann', 1.0, fcs, tc.p.fs, f2s, s2f);
            a_w = 4 * ones(numel(fcs), 1);
            c   = filterbank(tc.sig.noise_mono, gs, a_w);
            tc.verifyEqual(numel(c), numel(fcs), ...
                'filterbank + warpedblfilter: coefficient count must match filter count.');
        end

        function testWarpedblfilterDifferentWindowShapes(tc)
            f2s     = @(f) freqtoaud(f, 'erb');
            s2f     = @(s) audtofreq(s, 'erb');
            windows = {'hann', 'hamming', 'blackman'};
            for k = 1 : numel(windows)
                g = warpedblfilter(windows{k}, 1.0, 500, tc.p.fs, f2s, s2f);
                tc.verifyTrue(isstruct(g), ...
                    sprintf('warpedblfilter(''%s'',...): must return a struct.', windows{k}));
            end
        end

    end

    % ── filterbankfreqz ───────────────────────────────────────────────────
    methods (Test)

        function testFilterbankfreqzOutputSize(tc)
            gf = filterbankfreqz(tc.g, tc.a, tc.L);
            tc.verifyEqual(size(gf, 1), tc.L, ...
                'filterbankfreqz: row count must equal L.');
            tc.verifyEqual(size(gf, 2), tc.M, ...
                'filterbankfreqz: column count must equal M (number of filters).');
        end

        function testFilterbankfreqzSingleFilter(tc)
            gf = filterbankfreqz(tc.g(1), tc.a(1,:), tc.L);
            tc.verifyEqual(size(gf, 1), tc.L, ...
                'filterbankfreqz (single): row count must equal L.');
            tc.verifyEqual(size(gf, 2), 1, ...
                'filterbankfreqz (single): must return exactly 1 column.');
        end

        function testFilterbankfreqzSumSquaredMatchesFiltbankresponse(tc)
            % sum_m |gf(:,m)|^2 / a_m  must equal filterbankresponse(g, a, L).
            gf       = filterbankfreqz(tc.g, tc.a, tc.L);
            a_vals   = tc.a(:, 1);
            gf2_sum  = zeros(tc.L, 1);
            for m = 1 : tc.M
                gf2_sum = gf2_sum + abs(gf(:, m)).^2 / a_vals(m);
            end
            gf2_ref = real(filterbankresponse(tc.g, tc.a, tc.L));
            relErr  = norm(gf2_sum - gf2_ref) / (norm(gf2_ref) + eps);
            tc.verifyLessThan(relErr, 1e-8, ...
                'filterbankfreqz: squared-magnitude sum must match filterbankresponse.');
        end

        function testFilterbankfreqzRealFIRHermitianSymmetry(tc)
            % For real FIR filters, gf(L-k+2,:) == conj(gf(k,:)).
            M_fir = 4;
            g_fir = cell(1, M_fir);
            for m = 1 : M_fir
                g_fir{m} = firfilter('hann', 32);
            end
            a_fir = 8 * ones(M_fir, 1);
            L_fir = filterbanklength(tc.p.Ls, a_fir);
            gf    = filterbankfreqz(g_fir, a_fir, L_fir);
            % Hermitian symmetry: bin 2 == conj(bin L) for a real filter.
            if L_fir >= 4
                err = norm(gf(2,:) - conj(gf(L_fir,:)));
                tc.verifyLessThan(err, 1e-10, ...
                    'filterbankfreqz: real FIR responses must be Hermitian-symmetric.');
            end
        end

        function testFilterbankfreqzCqtFilters(tc)
            [g_cqt, a_cqt, ~, L_cqt] = cqtfilters(tc.p.fs, 100, 3000, 12, tc.p.Ls);
            gf = filterbankfreqz(g_cqt, a_cqt, L_cqt);
            tc.verifyEqual(size(gf, 1), L_cqt, ...
                'filterbankfreqz (CQT): row count must equal L.');
            tc.verifyEqual(size(gf, 2), numel(g_cqt), ...
                'filterbankfreqz (CQT): column count must equal numel(g).');
        end

    end

end
