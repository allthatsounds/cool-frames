classdef TestFreqFilter < matlab.unittest.TestCase
%TESTFREQFILTER  Unit tests for freqfilter (frequency-domain filter constructor).
%
%   freqfilter(name, bw)          — Gaussian/Gammatone/Butterworth bandpass at DC
%   freqfilter(name, bw, fc)      — same, shifted to centre freq fc (normalized)

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
            g = freqfilter('gauss', 0.1);
            tc.verifyTrue(isfield(g, 'H'),        'Missing: H');
            tc.verifyTrue(isfield(g, 'foff'),     'Missing: foff');
            tc.verifyTrue(isfield(g, 'delay'),    'Missing: delay');
            tc.verifyTrue(isfield(g, 'realonly'), 'Missing: realonly');
        end

        function testHIsAFunctionHandle(tc)
            g = freqfilter('gauss', 0.1);
            tc.verifyTrue(isa(g.H, 'function_handle'), 'g.H must be a function handle');
        end

        function testScalarInputReturnsStruct(tc)
            g = freqfilter('gauss', 0.1);
            tc.verifyTrue(isstruct(g));
        end

        function testDefaultRealonlyIsZero(tc)
            g = freqfilter('gauss', 0.1, 0.3);
            tc.verifyEqual(g.realonly, 0);
        end

        function testRealFlagSetsRealonly(tc)
            g = freqfilter('gauss', 0.1, 0.3, 'real');
            tc.verifyEqual(g.realonly, 1);
        end

    end

    % ── Transfer function ─────────────────────────────────────────────────────
    methods (Test)

        function testTransferFunctionLength(tc)
            g = freqfilter('gauss', 0.1, 0.3);
            for L = [128, 256, 1024]
                H = comp_transferfunction(g, L);
                tc.verifyEqual(numel(H), L, ...
                    sprintf('comp_transferfunction: expected length %d', L));
            end
        end

        function testTransferFunctionFinite(tc)
            g = freqfilter('gauss', 0.1, 0.3);
            H = comp_transferfunction(g, 256);
            tc.verifyTrue(all(isfinite(H)), 'Transfer function has non-finite values');
        end

        function testTransferFunctionNotAllZero(tc)
            g = freqfilter('gauss', 0.1, 0.3);
            H = comp_transferfunction(g, 256);
            tc.verifyGreaterThan(max(abs(H)), 0);
        end

        function testGammatoneTransferFunction(tc)
            g = freqfilter('gammatone', 0.1, 0.3);
            H = comp_transferfunction(g, 256);
            tc.verifyEqual(numel(H), 256);
            tc.verifyTrue(all(isfinite(H)));
        end

        function testButterworthTransferFunction(tc)
            g = freqfilter('butterworth', 0.1, 0.3);
            H = comp_transferfunction(g, 256);
            tc.verifyEqual(numel(H), 256);
            tc.verifyTrue(all(isfinite(H)));
        end

    end

    % ── Energy normalization ──────────────────────────────────────────────────
    methods (Test)

        function testEnergyNormDefault(tc)
            % Default: 'energy' — (1/L)*sum|H|^2 = 1
            L = 512;
            g = freqfilter('gauss', 0.05, 0.3);
            H = comp_transferfunction(g, L);
            energy = sum(abs(H).^2) / L;
            tc.verifyEqual(energy, 1, 'AbsTol', tc.tol, ...
                'freqfilter energy norm: (1/L)*sum|H|^2 should equal 1');
        end

        function testPeakNorm(tc)
            L = 512;
            g = freqfilter('gauss', 0.05, 0.3, 'peak');
            H = comp_transferfunction(g, L);
            tc.verifyEqual(max(abs(H)), 1, 'AbsTol', tc.tol, ...
                'freqfilter peak norm: max|H| should equal 1');
        end

        function testScalParameter(tc)
            L  = 256;
            s  = 4.0;
            g1 = freqfilter('gauss', 0.05, 0.3);
            g2 = freqfilter('gauss', 0.05, 0.3, 'scal', s);
            H1 = comp_transferfunction(g1, L);
            H2 = comp_transferfunction(g2, L);
            tc.verifyEqual(H2, s * H1, 'AbsTol', tc.tol * norm(H1), ...
                '''scal'' should multiply response by constant');
        end

    end

    % ── Centre frequency ──────────────────────────────────────────────────────
    methods (Test)

        function testPeakNearCentreFreq(tc)
            L  = 1024;
            fc = 0.3;
            g  = freqfilter('gauss', 0.05, fc, 'peak');
            H  = comp_transferfunction(g, L);
            [~, peak_bin] = max(abs(H));
            expected_bin  = round(fc / 2 * L) + 1;
            tc.verifyLessThan(abs(peak_bin - expected_bin), 5, ...
                'freqfilter: peak should be near specified fc');
        end

        function testHzInputConsistency(tc)
            % When 'fs' is specified, both bw and fc are in Hz.
            % freqfilter converts via /fs internally.  The normalised call
            % must apply the same conversion: bw_norm = bw_hz/(fs/2),
            % fc_norm = fc_hz/(fs/2).
            fs    = 8000;
            fc_hz = 1000;
            bw_hz = 400;   % Hz; normalised equivalent = 400/(fs/2) = 0.1
            g_hz   = freqfilter('gauss', bw_hz,        fc_hz,        'fs', fs);
            g_norm = freqfilter('gauss', bw_hz/(fs/2),  fc_hz/(fs/2));
            L = 512;
            H_hz   = comp_transferfunction(g_hz,   L);
            H_norm = comp_transferfunction(g_norm,  L);
            tc.verifyEqual(H_hz, H_norm, 'AbsTol', tc.tol, ...
                'Hz and normalised inputs should give same filter');
        end

    end

    % ── Delay ─────────────────────────────────────────────────────────────────
    methods (Test)

        function testDefaultDelayIsZero(tc)
            g = freqfilter('gauss', 0.1, 0.3);
            tc.verifyEqual(g.delay, 0);
        end

        function testDelayParameterStored(tc)
            g = freqfilter('gauss', 0.1, 0.3, 'delay', 7);
            tc.verifyEqual(g.delay, 7);
        end

    end

end
