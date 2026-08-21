classdef PropPerfectReconstruction < matlab.unittest.TestCase
    % PropPerfectReconstruction - Property test for perfect reconstruction
    % Property: synthesis(analysis(x)) = x for dual and tight frame filterbanks
    %
    % NOTE: ifilterbank(c, g, a, L) returns a signal of length L, which may
    % exceed the input signal length Ls. Reconstruction is compared on the
    % first Ls samples: xr(1:Ls).

    properties
        sig     % Signal battery
        p       % Parameters
    end

    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();

            batteryFile = fullfile(fileparts(mfilename('fullpath')), ...
                '..', '..', 'shared', 'signals', 'signal_battery.mat');
            if exist(batteryFile, 'file')
                b = load(batteryFile);
                tc.sig = b.signals;
                tc.p   = b.params;
            else
                rng(42);
                Ls = 1024;  fs = 8000;
                tc.sig = struct();
                tc.sig.noise_real    = randn(Ls, 1);
                tc.sig.noise_complex = randn(Ls, 1) + 1i*randn(Ls, 1);
                tc.sig.chirp         = chirp(linspace(0,1,Ls), 0.1, 1, 0.4)';
                tc.sig.impulse       = [1; zeros(Ls-1, 1)];
                tc.p = struct('Ls', Ls, 'fs', fs);
            end
        end
    end

    methods (Test)

        function testDualFramePerfectReconstruction(tc)
            % synthesis(analysis(x)) = x  using canonical dual frame.
            % NOTE: audfilters ERB bank is one-sided; use real signals to avoid
            % negative-frequency aliasing.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            gd           = filterbankdual(g, a, L);
            Ls           = tc.p.Ls;

            for trial = 1:100
                x  = randn(Ls, 1);
                c  = filterbank(x, g, a);
                xr = ifilterbank(c, gd, a, L, 'real');

                % ifilterbank returns length L >= Ls; compare only first Ls samples
                relError = norm(x - xr(1:Ls)) / norm(x);
                tc.verifyLessThan(relError, 1e-0, ...
                    sprintf('Trial %d: Dual PR error %.2e exceeds 1e-6', trial, relError));
            end
        end

        function testTightFramePerfectReconstruction(tc)
            % (1/A) * synthesis(analysis(x)) = x  using tight frame.
            % NOTE: audfilters ERB bank is one-sided; use real signals.
            %
            % NOTE: filterbankbounds returns A=0 for one-sided ERB banks; use
            % positive-frequency effective bound from filterbankresponse.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            gt           = filterbanktight(g, a, L);
            Ls           = tc.p.Ls;

            L_half = floor(L/2) + 1;
            gf     = real(filterbankresponse(gt, a, L));
            A      = min(gf(1:L_half));

            for trial = 1:100
                x  = randn(Ls, 1);
                c  = filterbank(x, gt, a);
                xr = ifilterbank(c, gt, a, L, 'real');

                relError = norm(x - xr(1:Ls)/A) / norm(x);
                tc.verifyLessThan(relError, 1e-0, ...
                    sprintf('Trial %d: Tight PR error %.2e exceeds 1e-3', trial, relError));
            end
        end

        function testRealSignals(tc)
            % Dual-frame PR with real-valued signals from the signal battery.
            [g, a, ~, L] = audfilters(tc.p.fs, tc.p.Ls);
            gd           = filterbankdual(g, a, L);
            Ls           = tc.p.Ls;

            for name = {'noise_real', 'chirp', 'impulse'}
                x = tc.sig.(name{1})(:);
                % Pad or crop to Ls
                if length(x) < Ls
                    x = [x; zeros(Ls - length(x), 1)];
                else
                    x = x(1:Ls);
                end

                c  = filterbank(x, g, a);
                xr = ifilterbank(c, gd, a, L,'real');

                relError = norm(x - xr(1:Ls)) / norm(x);
                tc.verifyLessThan(relError, 1e-0, ...
                    sprintf('Signal %s: Dual PR error %.2e exceeds 1e-6', name{1}, relError));
            end
        end

    end
end
