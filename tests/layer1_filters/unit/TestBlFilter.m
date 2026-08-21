classdef TestBlFilter < matlab.unittest.TestCase
%TESTBLFILTER  Unit tests for blfilter (band-limited filter constructor).
%
%   blfilter(name, fsupp)         — BL filter centred at DC
%   blfilter(name, fsupp, fc)     — BL filter at centre freq fc (normalized)
%   blfilter(name, fsupp, fc, norm) — with explicit normalization

    properties
        tol = 1e-10
    end

    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
        end
    end

    % ── Struct format ────────────────────────────────────────────────────────
    methods (Test)

        function testStructHasRequiredFields(tc)
            g = blfilter('hann', 0.1);
            tc.verifyTrue(isfield(g, 'H'),        'Missing: H');
            tc.verifyTrue(isfield(g, 'foff'),     'Missing: foff');
            tc.verifyTrue(isfield(g, 'delay'),    'Missing: delay');
            tc.verifyTrue(isfield(g, 'realonly'), 'Missing: realonly');
        end

        function testHIsAFunctionHandle(tc)
            g = blfilter('hann', 0.1);
            tc.verifyTrue(isa(g.H, 'function_handle'), 'g.H must be a function handle');
        end

        function testFoffIsAFunctionHandle(tc)
            g = blfilter('hann', 0.1, 0.3);
            tc.verifyTrue(isa(g.foff, 'function_handle'), ...
                'g.foff must be a function handle for a shifted filter');
        end

        function testScalarInputReturnsStruct(tc)
            g = blfilter('hann', 0.1);
            tc.verifyTrue(isstruct(g));
        end

        function testDefaultRealonlyIsZero(tc)
            g = blfilter('hann', 0.1, 0.3);
            tc.verifyEqual(g.realonly, 0);
        end

        function testRealFlagSetsRealonly(tc)
            g = blfilter('hann', 0.1, 0.3, 'real');
            tc.verifyEqual(g.realonly, 1);
        end

    end

    % ── Transfer function via comp_transferfunction ──────────────────────────
    methods (Test)

        function testTransferFunctionLength(tc)
            g = blfilter('hann', 0.1, 0.3);
            for L = [128, 256, 1024]
                H = comp_transferfunction(g, L);
                tc.verifyEqual(numel(H), L, ...
                    sprintf('comp_transferfunction: expected length %d', L));
            end
        end

        function testTransferFunctionFinite(tc)
            g = blfilter('hann', 0.1, 0.3);
            H = comp_transferfunction(g, 256);
            tc.verifyTrue(all(isfinite(H)), 'Transfer function has non-finite values');
        end

        function testTransferFunctionNotAllZero(tc)
            g = blfilter('hann', 0.1, 0.3);
            H = comp_transferfunction(g, 256);
            tc.verifyGreaterThan(max(abs(H)), 0, ...
                'Transfer function should not be identically zero');
        end

    end

    % ── Energy normalization ─────────────────────────────────────────────────
    methods (Test)

        function testEnergyNormDefault(tc)
            % Default norm is 'energy': (1/L)*sum|H|^2 = 1
            L = 512;
            g = blfilter('hann', 0.1, 0.3);
            H = comp_transferfunction(g, L);
            energy = sum(abs(H).^2) / L;
            tc.verifyEqual(energy, 1, 'AbsTol', tc.tol, ...
                'Energy normalization: (1/L)*sum|H|^2 should equal 1');
        end

        function testEnergyNormExplicit(tc)
            L = 256;
            g = blfilter('hann', 0.1, 0.3, 'energy');
            H = comp_transferfunction(g, L);
            energy = sum(abs(H).^2) / L;
            tc.verifyEqual(energy, 1, 'AbsTol', tc.tol);
        end

        function testPeakNorm(tc)
            % 'peak': max|H| = 1
            L = 512;
            g = blfilter('hann', 0.1, 0.3, 'peak');
            H = comp_transferfunction(g, L);
            tc.verifyEqual(max(abs(H)), 1, 'AbsTol', tc.tol, ...
                'Peak normalization: max|H| should equal 1');
        end

        function testScalParameter(tc)
            % 'scal' multiplies the output by that constant
            L   = 256;
            s   = 3.0;
            g1  = blfilter('hann', 0.1, 0.3);
            g2  = blfilter('hann', 0.1, 0.3, 'scal', s);
            H1  = comp_transferfunction(g1, L);
            H2  = comp_transferfunction(g2, L);
            tc.verifyEqual(H2, s * H1, 'AbsTol', tc.tol * norm(H1), ...
                '''scal'' should multiply entire response');
        end

    end

    % ── Centre frequency ─────────────────────────────────────────────────────
    methods (Test)

        function testPeakNearCentreFreq(tc)
            % Peak of |H| should be within 3 bins of fc
            L  = 1024;
            fc = 0.3;   % normalized, fc=1 is Nyquist
            g  = blfilter('hann', 0.1, fc, 'peak');
            H  = comp_transferfunction(g, L);
            [~, peak_bin] = max(abs(H));
            expected_bin  = round(fc / 2 * L) + 1;   % 1-indexed
            tc.verifyLessThan(abs(peak_bin - expected_bin), 4, ...
                'blfilter: peak should be near specified fc');
        end

        function testDCFilterPeakAtBin1(tc)
            % fc = 0: filter centred at DC, peak at bin 1
            L = 256;
            g = blfilter('hann', 0.1, 0, 'peak');
            H = comp_transferfunction(g, L);
            [~, peak_bin] = max(abs(H));
            tc.verifyEqual(peak_bin, 1, ...
                'DC-centred filter: peak should be at bin 1');
        end

        function testHzInputConsistency(tc)
            % Specifying fc in Hz via 'fs' should match the normalised version.
            % bw_hz and fc_hz are in Hz; blfilter converts via /fs*2.
            % The normalised call must apply the same conversion manually.
            fs    = 8000;
            fc_hz = 1000;
            bw_hz = 400;   % Hz; normalised equivalent = 400/(fs/2) = 0.1
            g_hz   = blfilter('hann', bw_hz,        fc_hz,        'fs', fs);
            g_norm = blfilter('hann', bw_hz/(fs/2),  fc_hz/(fs/2));
            L = 512;
            H_hz   = comp_transferfunction(g_hz,   L);
            H_norm = comp_transferfunction(g_norm,  L);
            tc.verifyEqual(H_hz, H_norm, 'AbsTol', tc.tol, ...
                'Hz and normalised inputs should give same filter');
        end

    end

    % ── Delay field ──────────────────────────────────────────────────────────
    methods (Test)

        function testDefaultDelayIsZero(tc)
            g = blfilter('hann', 0.1, 0.3);
            tc.verifyEqual(g.delay, 0);
        end

        function testDelayParameterStored(tc)
            g = blfilter('hann', 0.1, 0.3, 'delay', 5);
            tc.verifyEqual(g.delay, 5);
        end

    end

end
