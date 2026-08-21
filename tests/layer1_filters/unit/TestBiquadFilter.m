classdef TestBiquadFilter < matlab.unittest.TestCase
%TESTBIQUADFILTER  Unit tests for biquadfilter and comp_biquad.
%
%   Tests the biquadfilter constructor and the underlying comp_biquad
%   primitive, verifying:
%     - Output struct has the required fields
%     - Pole parameters derived correctly from (fc, bw)
%     - Stability-preserving ML parametrization (rho, phi) is consistent
%     - Frequency response peaks at the correct bin
%     - Normalization modes produce the specified energy
%     - Filter integrates correctly with filterbank / ifilterbank
%     - Vector input yields cell-array output (matching blfilter behaviour)
%
%   See also: biquadfilter, comp_biquad, blfilter

    properties
        fs = 8000
        Ls = 1024
        tol_loose = 1e-6
        tol_tight = 1e-12
    end

    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
        end
    end

    % ── Output struct format ──────────────────────────────────────────────────
    methods (Test)

        function testStructHasRequiredFields(tc)
            g = biquadfilter(0.25, 0.05);
            tc.verifyTrue(isfield(g, 'H'),        'Missing field: H');
            tc.verifyTrue(isfield(g, 'foff'),     'Missing field: foff');
            tc.verifyTrue(isfield(g, 'delay'),    'Missing field: delay');
            tc.verifyTrue(isfield(g, 'realonly'), 'Missing field: realonly');
            tc.verifyTrue(isfield(g, 'r'),        'Missing field: r');
            tc.verifyTrue(isfield(g, 'theta'),    'Missing field: theta');
            tc.verifyTrue(isfield(g, 'rho'),      'Missing field: rho');
            tc.verifyTrue(isfield(g, 'phi'),      'Missing field: phi');
        end

        function testHIsAFunctionHandle(tc)
            g = biquadfilter(0.25, 0.05);
            tc.verifyTrue(isa(g.H, 'function_handle'), 'g.H must be a function handle');
        end

        function testFoffIsZero(tc)
            % Full-length IIR response: foff must be a function handle that
            % always evaluates to 0 (required by comp_fourierwindow when
            % g.H is a function handle, same convention as blfilter).
            g = biquadfilter(0.25, 0.05);
            tc.verifyTrue(isa(g.foff, 'function_handle'), ...
                'g.foff must be a function handle');
            tc.verifyEqual(g.foff(1), 0,   'g.foff(1) must equal 0');
            tc.verifyEqual(g.foff(512), 0, 'g.foff(512) must equal 0');
        end

        function testHReturnsCorrectLength(tc)
            g = biquadfilter(0.25, 0.05);
            for L = [64, 256, 1024]
                H = g.H(L);
                tc.verifyEqual(numel(H), L, ...
                    sprintf('H must have length L=%d', L));
            end
        end

        function testVectorInputYieldsCellArray(tc)
            fc_vec = [0.1, 0.2, 0.3];
            bw_vec = [0.05, 0.05, 0.05];
            gout = biquadfilter(fc_vec, bw_vec);
            tc.verifyTrue(iscell(gout), 'Vector input must produce cell array');
            tc.verifyEqual(numel(gout), 3);
            for k = 1:3
                tc.verifyTrue(isfield(gout{k}, 'H'));
            end
        end

        function testScalarInputReturnsStruct(tc)
            g = biquadfilter(0.25, 0.05);
            tc.verifyTrue(isstruct(g), 'Scalar input must return struct (not cell)');
        end

    end

    % ── Pole parameter correctness ────────────────────────────────────────────
    methods (Test)

        function testPoleRadiusFromBandwidth(tc)
            % r ≈ 1 - pi*bw/2  (exact only in the limit r→1)
            bw  = 0.04;
            g   = biquadfilter(0.25, bw);
            r_expected = 1 - pi*bw/2;
            tc.verifyLessThan(abs(g.r - r_expected), 1e-10);
        end

        function testPoleAngleFromCenterFreq(tc)
            % theta = pi * |fc|
            fc = 0.3;
            g  = biquadfilter(fc, 0.05);
            tc.verifyLessThan(abs(g.theta - pi*fc), 1e-12);
        end

        function testStabilityConstraint(tc)
            % Pole radius must always be strictly in (0, 1)
            for bw = [0.001, 0.01, 0.1, 0.5]
                g = biquadfilter(0.25, bw);
                tc.verifyGreaterThan(g.r, 0, 'r must be > 0');
                tc.verifyLessThan(g.r, 1,   'r must be < 1 (stable)');
            end
        end

        function testMLParametrizationConsistency(tc)
            % rho = logit(r), phi = logit(theta/pi) — verify roundtrip
            g = biquadfilter(0.3, 0.04);

            r_from_rho     = 1 / (1 + exp(-g.rho));
            theta_from_phi = pi / (1 + exp(-g.phi));

            tc.verifyLessThan(abs(r_from_rho     - g.r),     1e-12);
            tc.verifyLessThan(abs(theta_from_phi - g.theta), 1e-12);
        end

        function testMLOverrideRho(tc)
            % Passing 'rho' overrides the bw-derived radius
            rho_in = 2.0;
            r_expected = 1 / (1 + exp(-rho_in));
            g = biquadfilter(0.25, 0.05, 'rho', rho_in);
            tc.verifyLessThan(abs(g.r - r_expected), 1e-12);
        end

        function testMLOverridePhi(tc)
            % Passing 'phi' overrides the fc-derived angle
            phi_in   = 0.5;
            th_expected = pi / (1 + exp(-phi_in));
            g = biquadfilter(0.25, 0.05, 'phi', phi_in);
            tc.verifyLessThan(abs(g.theta - th_expected), 1e-12);
        end

    end

    % ── Frequency response properties ─────────────────────────────────────────
    methods (Test)

        function testFreqResponsePeaksNearCenterFreq(tc)
            % The peak of |H| should occur at the bin closest to fc
            L  = 1024;
            fc = 0.3;
            g  = biquadfilter(fc, 0.02, 'peak');
            H  = g.H(L);
            [~, peak_bin] = max(abs(H));

            % Expected peak bin (LTFAT: fc=1 → Nyquist → bin L/2)
            expected_bin = round(fc / 2 * L) + 1;   % 1-indexed
            % Allow ±2 bins for rounding
            tc.verifyLessThan(abs(peak_bin - expected_bin), 3, ...
                'Peak should be within 2 bins of fc');
        end

        function testEnergyNormalization(tc)
            % 'energy' normalization: (1/L)*sum|H|^2 = 1  <=>  ||H||/sqrt(L) = 1
            L = 512;
            g = biquadfilter(0.25, 0.03, 'energy');
            H = g.H(L);
            energy = sum(abs(H).^2) / L;
            tc.verifyLessThan(abs(energy - 1), tc.tol_loose, ...
                'Energy normalization should give (1/L)*sum|H|^2 = 1');
        end

        function testPeakNormalization(tc)
            % 'peak' normalization: max|H| = 1
            L = 512;
            g = biquadfilter(0.25, 0.03, 'peak');
            H = g.H(L);
            tc.verifyLessThan(abs(max(abs(H)) - 1), tc.tol_loose, ...
                'Peak normalization should give max|H| = 1');
        end

        function testScalParameter(tc)
            % 'scal' scales the entire response by a constant
            L    = 256;
            s    = 2.5;
            g1   = biquadfilter(0.25, 0.03);
            g2   = biquadfilter(0.25, 0.03, 'scal', s);
            H1   = g1.H(L);
            H2   = g2.H(L);
            tc.verifyLessThan(max(abs(H2 - s*H1)), tc.tol_loose * norm(H1));
        end

        function testHzInputConsistency(tc)
            % Hz and normalized inputs should produce the same filter
            fs  = tc.fs;
            fc_hz = 1000;
            bw_hz = 200;
            g_hz   = biquadfilter(fc_hz, bw_hz, 'fs', fs);
            g_norm = biquadfilter(fc_hz/(fs/2), bw_hz/(fs/2));

            L = 512;
            H_hz   = g_hz.H(L);
            H_norm = g_norm.H(L);
            tc.verifyLessThan(max(abs(H_hz - H_norm)), tc.tol_loose);
        end

        function testRealOnlyFlag(tc)
            % 'real' flag sets g.realonly = 1
            g = biquadfilter(0.25, 0.05, 'real');
            tc.verifyEqual(g.realonly, 1);
        end

        function testComplexDefault(tc)
            % Default 'complex' flag: g.realonly = 0
            g = biquadfilter(0.25, 0.05);
            tc.verifyEqual(g.realonly, 0);
        end

        function testDelayParameter(tc)
            g = biquadfilter(0.25, 0.05, 'delay', 3);
            tc.verifyEqual(g.delay, 3);
        end

    end

    % ── Filterbank integration ────────────────────────────────────────────────
    methods (Test)

        function testIntegrationWithFilterbank(tc)
            % biquadfilter must work as a drop-in for filterbank()
            rng(42);
            x  = randn(tc.Ls, 1);
            fs = tc.fs;
            Ls = tc.Ls;

            % Build a small cell array of biquad filters
            fc_vec = [0.1, 0.25, 0.4];
            bw_val = 0.05;
            a      = [4; 4; 4];    % integer subsampling

            g = cell(1, numel(fc_vec));
            for k = 1:numel(fc_vec)
                g{k} = biquadfilter(fc_vec(k), bw_val);
            end

            % Analysis
            c = filterbank(x, g, a);
            tc.verifyEqual(numel(c), numel(fc_vec));
            for k = 1:numel(fc_vec)
                tc.verifyEqual(size(c{k}, 1), ceil(Ls / a(k)));
            end
        end

        function testZeroInputYearsZeroOutput(tc)
            % Zero signal → zero subbands
            x  = zeros(tc.Ls, 1);
            g  = {biquadfilter(0.25, 0.05)};
            a  = 2;
            c  = filterbank(x, g, a);
            tc.verifyLessThan(max(abs(c{1})), tc.tol_tight, ...
                'Zero input must give zero subband output');
        end

    end

    % ── comp_biquad directly ──────────────────────────────────────────────────
    methods (Test)

        function testCompBiquadOutputLength(tc)
            for L = [32, 64, 512]
                H = comp_biquad(0.9, pi/4, L, 'energy');
                tc.verifyEqual(numel(H), L);
            end
        end

        function testCompBiquadPoleAtUnityIsUnstable(tc)
            % r = 1 → D has a zero on the unit circle → H has a pole
            % The response at that frequency should be very large
            L = 128;
            r = 1 - 1e-6;   % approach the boundary
            H = comp_biquad(r, pi/4, L, 'inf');
            tc.verifyGreaterThan(max(abs(H)), 100, ...
                'Near-marginal filter should have large peak');
        end

        function testCompBiquadStabilityAcrossParameters(tc)
            % For any r < 1 and any theta, the response must be finite
            rng(42);
            for trial = 1:50
                r     = rand() * 0.99;
                theta = rand() * pi;
                H     = comp_biquad(r, theta, 128, 'inf');
                tc.verifyTrue(all(isfinite(H)), ...
                    sprintf('Infinite H at r=%.3f theta=%.3f', r, theta));
            end
        end

        function testCompBiquadEnergyNorm(tc)
            % Verify energy normalization formula for several (r, theta)
            rng(42);
            for trial = 1:20
                r     = 0.5 + rand()*0.4;
                theta = rand()*pi;
                L     = 256;
                H     = comp_biquad(r, theta, L, 'energy');
                energy = sum(abs(H).^2) / L;
                tc.verifyLessThan(abs(energy - 1), tc.tol_loose, ...
                    'Energy normalization failed');
            end
        end

    end

end
